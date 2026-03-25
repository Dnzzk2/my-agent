import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { stdin as input, stdout as output } from "node:process";
import readline from "node:readline/promises";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config({ override: true });

const WORKDIR = process.cwd();
const MODEL = getRequiredEnv("MODEL_ID");
const SYSTEM = `You are a coding agent at ${WORKDIR}. Use tools to solve tasks.`;
const THRESHOLD = 6000;
const TRANSCRIPT_DIR = path.join(WORKDIR, ".transcripts");
const KEEP_RECENT = 3;
const TOOL_OUTPUT_LIMIT = 50000;
const DEFAULT_MAX_TOKENS = 8000;
const SUMMARY_MAX_TOKENS = 2000;

// 这里使用 OpenAI 官方 SDK，但 baseURL/apiKey 默认指向 DeepSeek。
// 原因是 DeepSeek 提供了 OpenAI 兼容接口，所以可以直接复用同一套 SDK 和消息格式。
//
// 可以把它理解成：
// - “客户端库”是 OpenAI 的
// - “服务端实现”默认是 DeepSeek 的兼容接口
// 只要接口协议兼容，这种组合就是成立的。
const openai = new OpenAI({
	baseURL: process.env.DEEPSEEK_BASE_URL ?? process.env.OPENAI_BASE_URL,
	apiKey:
		process.env.DEEPSEEK_API_KEY ??
		process.env.OPENAI_API_KEY ??
		process.env.API_KEY,
});

// 整个学习项目统一使用 OpenAI 语义的消息结构：
// - assistant 通过 tool_calls 发起工具调用
// - tool 通过 tool_call_id 把结果回传给模型
// 这样读代码时不需要在 Anthropic / OpenAI 两套术语之间来回切换。
//
// 一次完整的工具调用链路长这样：
// 1. user       -> 提交自然语言问题
// 2. assistant  -> 返回 tool_calls，要求执行某个工具
// 3. tool       -> 使用同一个 tool_call_id 回传工具结果
// 4. assistant  -> 基于工具结果继续思考，可能再次发起 tool_calls，也可能直接回答
type UserMessage = {
	role: "user";
	content: string;
};

// tool 消息不会主动出现，它一定是“响应某个 assistant.tool_calls”而产生的。
// tool_call_id 的作用就是把“这条工具结果”绑定回“当时那次工具调用”。
type ToolMessage = {
	role: "tool";
	tool_call_id: string;
	content: string;
};

// assistant.tool_calls 是模型告诉宿主程序：
// “我现在不直接回答，请你先替我执行这些函数，再把结果发回来。”
type AssistantToolCall = {
	id: string;
	type: "function";
	function: {
		name: string;
		arguments: string;
	};
};

type AssistantMessage = {
	role: "assistant";
	content: string | null;
	tool_calls?: AssistantToolCall[];
};

type ConversationMessage = UserMessage | ToolMessage | AssistantMessage;

type ToolDefinition = {
	name: string;
	description: string;
	input_schema: {
		type: "object";
		properties: Record<string, unknown>;
		required?: string[];
	};
};

type CompletionResult = {
	message: AssistantMessage;
	finish_reason: string | null;
};

function getToolCalls(message: AssistantMessage): AssistantToolCall[] {
	return message.tool_calls ?? [];
}

const TOOLS: ToolDefinition[] = [
	{
		name: "bash",
		description: "Run a shell command.",
		input_schema: {
			type: "object",
			properties: {
				command: { type: "string" },
			},
			required: ["command"],
		},
	},
	{
		name: "read_file",
		description: "Read file contents.",
		input_schema: {
			type: "object",
			properties: {
				path: { type: "string" },
				limit: { type: "integer" },
			},
			required: ["path"],
		},
	},
	{
		name: "write_file",
		description: "Write content to file.",
		input_schema: {
			type: "object",
			properties: {
				path: { type: "string" },
				content: { type: "string" },
			},
			required: ["path", "content"],
		},
	},
	{
		name: "edit_file",
		description: "Replace exact text in file.",
		input_schema: {
			type: "object",
			properties: {
				path: { type: "string" },
				old_text: { type: "string" },
				new_text: { type: "string" },
			},
			required: ["path", "old_text", "new_text"],
		},
	},
	{
		name: "compact",
		description: "Trigger manual conversation compression.",
		input_schema: {
			type: "object",
			properties: {
				focus: {
					type: "string",
					description: "What to preserve in the summary",
				},
			},
		},
	},
];

function getRequiredEnv(name: string): string {
	const value = process.env[name];
	if (!value) {
		throw new Error(`Missing required environment variable: ${name}`);
	}
	return value;
}

function estimateTokens(messages: ConversationMessage[]): number {
	return Math.floor(JSON.stringify(messages).length / 4);
}

// Layer 1: 每轮请求前先“轻压缩”旧的工具输出。
// 只保留最近 3 次 tool 消息的完整内容，更早的工具结果替换成简短占位符。
// 这样能显著降低上下文膨胀，但仍保留“当时调用过哪个工具”的线索。
//
// 注意这里不是随便删历史，而是做“有线索的遗忘”：
// - 旧的 tool.content 被缩短
// - 但 tool_call_id 还在
// - 同时我们还能回查当时 assistant.tool_calls 里的函数名
// 所以模型仍能知道：“之前调用过 read_file / bash / edit_file”，只是看不到完整输出了。
function microCompact(messages: ConversationMessage[]): ConversationMessage[] {
	const toolMessages = messages.filter(
		(message): message is ToolMessage => message.role === "tool",
	);

	if (toolMessages.length <= KEEP_RECENT) {
		return messages;
	}

	const toolNameMap = new Map<string, string>();
	for (const message of messages) {
		if (message.role !== "assistant") {
			continue;
		}

		for (const toolCall of message.tool_calls ?? []) {
			toolNameMap.set(toolCall.id, toolCall.function.name);
		}
	}

	// 这里通过 tool_call_id <-> assistant.tool_calls.id 做关联，
	// 从而把占位符写成 “[Previous: used xxx]”。
	for (const message of toolMessages.slice(0, -KEEP_RECENT)) {
		if (message.content.length <= 100) {
			continue;
		}

		const toolName = toolNameMap.get(message.tool_call_id) ?? "unknown";
		message.content = `[Previous: used ${toolName}]`;
	}

	return messages;
}

// Layer 2: 当上下文超过阈值时，先把完整历史落盘，再让模型做一份“连续性摘要”。
// 摘要完成后，直接用两条消息替换整个历史：
// 1) 用户侧保存摘要文本
// 2) 助手侧确认“我已加载压缩后的上下文”
async function autoCompact(
	messages: ConversationMessage[],
): Promise<ConversationMessage[]> {
	await fs.mkdir(TRANSCRIPT_DIR, { recursive: true });
	const transcriptPath = path.join(
		TRANSCRIPT_DIR,
		`transcript_${Math.floor(Date.now() / 1000)}.jsonl`,
	);

	const transcriptLines = messages
		.map((message) => `${JSON.stringify(message)}\n`)
		.join("");
	await fs.writeFile(transcriptPath, transcriptLines, "utf8");
	console.log(`[transcript saved: ${transcriptPath}]`);

	const conversationText = JSON.stringify(messages).slice(0, 80000);
	const response = await createCompletion({
		model: MODEL,
		messages: [
			{
				role: "user",
				content:
					"Summarize this conversation for continuity. Include: " +
					"1) What was accomplished, 2) Current state, 3) Key decisions made. " +
					"Be concise but preserve critical details.\n\n" +
					conversationText,
			},
		],
		max_tokens: SUMMARY_MAX_TOKENS,
	});

	const summary = response.message.content?.trim() ?? "";

	return [
		{
			role: "user",
			content: `[Conversation compressed. Transcript: ${transcriptPath}]\n\n${summary}`,
		},
		{
			role: "assistant",
			content: "Understood. I have the context from the summary. Continuing.",
		},
	];
}

// 所有文件工具都先经过 safePath。
// 目标是防止模型把路径写成 ../something，越过当前工作目录。
// 对学习项目来说，这一步很重要，因为它清楚展示了“工具能力必须加边界”。
function safePath(relativePath: string): string {
	const resolvedPath = path.resolve(WORKDIR, relativePath);
	const relativeToWorkdir = path.relative(WORKDIR, resolvedPath);

	if (
		relativeToWorkdir.startsWith("..") ||
		path.isAbsolute(relativeToWorkdir)
	) {
		throw new Error(`Path escapes workspace: ${relativePath}`);
	}

	return resolvedPath;
}

function truncateOutput(content: string): string {
	return content.length > TOOL_OUTPUT_LIMIT
		? content.slice(0, TOOL_OUTPUT_LIMIT)
		: content;
}

// bash 工具的职责很单纯：
// - 在工作目录执行命令
// - 截断超长输出
// - 屏蔽少数明显危险的命令片段
// 它不是完整沙箱，只是最基础的一层保护。
function runBash(command: string): string {
	const dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
	if (dangerous.some((item) => command.includes(item))) {
		return "Error: Dangerous command blocked";
	}

	const result = spawnSync(command, {
		shell: true,
		cwd: WORKDIR,
		encoding: "utf8",
		timeout: 120000,
		maxBuffer: 10 * 1024 * 1024,
	});

	if (result.error?.name === "TimeoutError") {
		return "Error: Timeout (120s)";
	}

	const output = `${result.stdout ?? ""}${result.stderr ?? ""}`.trim();
	if (result.error) {
		return truncateOutput(`Error: ${result.error.message}`);
	}

	return output.length > 0 ? truncateOutput(output) : "(no output)";
}

async function runRead(filePath: string, limit?: number): Promise<string> {
	try {
		const lines = (await fs.readFile(safePath(filePath), "utf8")).split(
			/\r?\n/,
		);
		if (typeof limit === "number" && limit < lines.length) {
			const clipped = lines.slice(0, limit);
			clipped.push(`... (${lines.length - limit} more)`);
			return truncateOutput(clipped.join("\n"));
		}
		return truncateOutput(lines.join("\n"));
	} catch (error) {
		return `Error: ${(error as Error).message}`;
	}
}

async function runWrite(filePath: string, content: string): Promise<string> {
	try {
		const resolvedPath = safePath(filePath);
		await fs.mkdir(path.dirname(resolvedPath), { recursive: true });
		await fs.writeFile(resolvedPath, content, "utf8");
		return `Wrote ${content.length} bytes`;
	} catch (error) {
		return `Error: ${(error as Error).message}`;
	}
}

async function runEdit(
	filePath: string,
	oldText: string,
	newText: string,
): Promise<string> {
	try {
		const resolvedPath = safePath(filePath);
		const content = await fs.readFile(resolvedPath, "utf8");
		if (!content.includes(oldText)) {
			return `Error: Text not found in ${filePath}`;
		}

		const matchIndex = content.indexOf(oldText);
		const nextContent =
			content.slice(0, matchIndex) +
			newText +
			content.slice(matchIndex + oldText.length);

		await fs.writeFile(resolvedPath, nextContent, "utf8");
		return `Edited ${filePath}`;
	} catch (error) {
		return `Error: ${(error as Error).message}`;
	}
}

const TOOL_HANDLERS: Record<
	string,
	(input: Record<string, unknown>) => string | Promise<string>
> = {
	bash: async (input) => runBash(String(input.command ?? "")),
	read_file: async (input) =>
		runRead(
			String(input.path ?? ""),
			typeof input.limit === "number" ? input.limit : undefined,
		),
	write_file: async (input) =>
		runWrite(String(input.path ?? ""), String(input.content ?? "")),
	edit_file: async (input) =>
		runEdit(
			String(input.path ?? ""),
			String(input.old_text ?? ""),
			String(input.new_text ?? ""),
		),
	compact: async () => "Manual compression requested.",
};

// ToolDefinition 是我们自己定义的“工具描述结构”；
// 这里只有在真正发请求前，才把它翻译成 OpenAI SDK 需要的工具格式。
function toOpenAITool(tool: ToolDefinition): OpenAI.Chat.ChatCompletionTool {
	return {
		type: "function",
		function: {
			name: tool.name,
			description: tool.description,
			parameters: tool.input_schema,
		},
	};
}

// 把内部历史消息转换成 OpenAI Chat Completions 请求格式。
// 这个函数只做“结构翻译”，不做业务逻辑判断。
//
// 三种消息角色分别对应：
// - user      -> 普通用户文本
// - assistant -> 模型上一轮的回答，可能携带 tool_calls
// - tool      -> 某个工具的返回值，必须带 tool_call_id
function toOpenAIMessages(
	messages: ConversationMessage[],
	system?: string,
): OpenAI.Chat.ChatCompletionMessageParam[] {
	const openaiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

	if (system) {
		openaiMessages.push({ role: "system", content: system });
	}

	for (const message of messages) {
		if (message.role === "user") {
			openaiMessages.push({
				role: "user",
				content: message.content,
			});
			continue;
		}

		if (message.role === "tool") {
			openaiMessages.push({
				role: "tool",
				tool_call_id: message.tool_call_id,
				content: message.content,
			});
			continue;
		}

		openaiMessages.push({
			role: "assistant",
			content: message.content,
			tool_calls: message.tool_calls,
		});
	}

	return openaiMessages;
}

function assertToolCallResponses(messages: ConversationMessage[]): void {
	for (let index = 0; index < messages.length; index += 1) {
		const message = messages[index];
		if (message.role !== "assistant") {
			continue;
		}

		const toolCalls = getToolCalls(message);
		if (toolCalls.length === 0) {
			continue;
		}

		const pendingIds = new Set(toolCalls.map((toolCall) => toolCall.id));
		let cursor = index + 1;

		while (cursor < messages.length && pendingIds.size > 0) {
			const nextMessage = messages[cursor];
			if (nextMessage.role !== "tool") {
				break;
			}

			pendingIds.delete(nextMessage.tool_call_id);
			cursor += 1;
		}

		if (pendingIds.size > 0) {
			throw new Error(
				`Invalid conversation history: missing tool responses for ${Array.from(pendingIds).join(", ")}`,
			);
		}
	}
}

// 把 OpenAI SDK 返回的 assistant message 收敛回我们的内部结构。
// 这样后面的 compact、日志、主循环都只处理一种统一格式，
// 不需要每个地方都直接耦合 SDK 的原始返回类型。
function fromOpenAIMessage(
	message: OpenAI.Chat.ChatCompletionMessage,
): AssistantMessage {
	const toolCalls: AssistantToolCall[] = [];
	for (const toolCall of message.tool_calls ?? []) {
		if (toolCall.type !== "function") {
			continue;
		}

		toolCalls.push({
			id: toolCall.id,
			type: "function",
			function: {
				name: toolCall.function.name,
				arguments: toolCall.function.arguments,
			},
		});
	}

	return {
		role: "assistant",
		content: typeof message.content === "string" ? message.content : null,
		tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
	};
}

// tool_calls 里的 arguments 是 JSON 字符串，不是对象。
// 所以执行工具前必须先 parse；解析失败时返回空对象，避免整轮流程崩掉。
function parseToolArguments(serializedArgs: string): Record<string, unknown> {
	try {
		const parsed = JSON.parse(serializedArgs);
		return typeof parsed === "object" && parsed !== null
			? (parsed as Record<string, unknown>)
			: {};
	} catch {
		return {};
	}
}

// createCompletion 是“模型调用边界”：
// - 输入：内部消息结构
// - 输出：内部 assistant 消息 + finish_reason
// 它把 SDK 细节关在这个函数里，外层只关心“模型说了什么、要不要继续调工具”。
async function createCompletion(body: {
	model: string;
	messages: ConversationMessage[];
	max_tokens: number;
	system?: string;
	tools?: ToolDefinition[];
}): Promise<CompletionResult> {
	assertToolCallResponses(body.messages);

	const response = await openai.chat.completions.create({
		model: body.model,
		messages: toOpenAIMessages(body.messages, body.system),
		tools: body.tools?.map(toOpenAITool),
		max_completion_tokens: body.max_tokens,
	});

	const choice = response.choices[0];
	return {
		message: fromOpenAIMessage(choice.message),
		finish_reason: choice.finish_reason,
	};
}

// 主循环只做三件事：
// 1) 请求模型
// 2) 执行模型要求的工具
// 3) 把工具结果按 tool 消息追加回历史
// 一旦 assistant 消息里不再携带 tool_calls，本轮就结束。
//
// 可以按下面的顺序理解这个循环：
// A. 先做 microCompact，避免每一轮都带着过长的旧工具输出
// B. 如果上下文还是太大，就触发 autoCompact，直接把长历史折叠成摘要
// C. 调模型
// D. 如果模型要求工具，就逐个执行，再把结果 append 成 tool 消息
// E. 如果模型没有要求工具，说明本轮回答结束，函数 return
async function agentLoop(messages: ConversationMessage[]): Promise<void> {
	while (true) {
		microCompact(messages);

		if (estimateTokens(messages) > THRESHOLD) {
			console.log("[auto_compact triggered]");
			const compactedMessages = await autoCompact(messages);
			messages.splice(0, messages.length, ...compactedMessages);
		}

		const response = await createCompletion({
			model: MODEL,
			system: SYSTEM,
			messages,
			tools: TOOLS,
			max_tokens: DEFAULT_MAX_TOKENS,
		});

		messages.push(response.message);

		const toolCalls = getToolCalls(response.message);
		if (toolCalls.length === 0) {
			return;
		}

		// 只要当前 assistant 消息里带着 tool_calls，
		// 我们就必须把这些工具逐个执行完，再把结果回传给下一轮模型。
		let manualCompact = false;
		for (const toolCall of toolCalls) {
			let output = "";

			if (toolCall.function.name === "compact") {
				manualCompact = true;
				output = "Compressing...";
			} else {
				const handler = TOOL_HANDLERS[toolCall.function.name];
				try {
					output = handler
						? await handler(parseToolArguments(toolCall.function.arguments))
						: `Unknown tool: ${toolCall.function.name}`;
				} catch (error) {
					output = `Error: ${(error as Error).message}`;
				}
			}

			console.log(
				`> ${toolCall.function.name}: ${String(output).slice(0, 200)}`,
			);
			messages.push({
				role: "tool",
				tool_call_id: toolCall.id,
				content: String(output),
			});
		}

		// Layer 3: 模型显式调用 compact 工具时，立刻做整段摘要压缩。
		if (manualCompact) {
			console.log("[manual compact]");
			const compactedMessages = await autoCompact(messages);
			messages.splice(0, messages.length, ...compactedMessages);
		}
	}
}

// main 只负责 CLI 交互：
// - 读取用户输入
// - 把输入写入 history
// - 调 agentLoop 跑完整的一轮“模型 <-> 工具”往返
// - 打印最后一条 assistant 文本
async function main(): Promise<void> {
	const history: ConversationMessage[] = [];
	const rl = readline.createInterface({ input, output });

	try {
		while (true) {
			let query = "";
			try {
				query = await rl.question("\u001b[36ms06 >> \u001b[0m");
			} catch (error) {
				if ((error as { code?: string }).code === "ABORT_ERR") {
					console.log();
					break;
				}
				throw error;
			}
			const normalized = query.trim().toLowerCase();

			if (normalized === "" || normalized === "q" || normalized === "exit") {
				break;
			}

			history.push({ role: "user", content: query });
			await agentLoop(history);

			const lastMessage = history.at(-1);
			if (lastMessage?.role === "assistant" && lastMessage.content) {
				console.log(lastMessage.content);
			}
			console.log();
		}
	} finally {
		rl.close();
	}
}

main().catch((error) => {
	if ((error as { code?: string }).code === "ABORT_ERR") {
		process.exitCode = 0;
		return;
	}
	console.error(error);
	process.exitCode = 1;
});
