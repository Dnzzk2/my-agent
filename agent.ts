/*
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> | Tools   |
| prompt |      |       |      | + todo  |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                          |
              +-----------+-----------+
              | TodoManager state     |
              | [ ] task A            |
              | [>] task B  <- doing  |
              | [x] task C            |
              +-----------------------+
                          |
              if rounds_since_todo >= 3:
                inject <reminder> into tool_result

*/

import { execSync } from "node:child_process";
import OpenAI from "openai";
import "dotenv/config";
import * as fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

/**
 * 全局常量配置
 */
const WORKDIR = process.cwd(); // 当前工作目录
const MODEL = "deepseek-chat"; // 使用的模型 ID
// 父智能体系统提示词：引导其使用 task 工具分发任务
const SYSTEM = `You are a coding agent at ${WORKDIR}. Use the task tool to delegate exploration or subtasks.`;
// 子智能体系统提示词：关注于完成具体任务并总结
const SUBAGENT_SYSTEM = `You are a coding subagent at ${WORKDIR}. Complete the given task, then summarize your findings.`;

const openai = new OpenAI({
	baseURL: process.env.DEEPSEEK_BASE_URL,
	apiKey: process.env.DEEPSEEK_API_KEY,
});

/**
 * 任务管理器类：负责维护和更新 Todo 列表
 */
class TodoManager {
	private items: { id: string; text: string; status: string }[] = [];
	/**
	 * 更新任务列表
	 * @param newItems 包含任务信息的数组，如 [{id, text, status}]
	 * @returns 返回渲染后的任务列表字符串
	 */
	update(newItems: any[]): string {
		if (newItems.length > 20) throw new Error("Max 20 todos allowed");

		let inProgressCount = 0;
		const validated = [];

		for (let i = 0; i < newItems.length; i++) {
			const item = newItems[i];
			const text = String(item.text || "").trim();
			const status = String(item.status || "pending").toLowerCase();
			const id = String(item.id || String(i + 1));

			if (!text) throw new Error(`Todo ${id} : must have text`);
			if (!["pending", "in_progress", "completed"].includes(status))
				throw new Error(`Todo ${id} : invalid status ${status}`);

			if (status === "in_progress") inProgressCount++;

			validated.push({ id, text, status });
		}

		// 约束：同一时间只能有一个任务处于正在处理中
		if (inProgressCount > 1)
			throw new Error("Only one todo can be in_progress at a time");
		this.items = validated;
		return this.render();
	}

	/**
	 * 渲染任务列表为易读的字符串格式
	 */
	render(): string {
		if (this.items.length === 0) return "No todo items.";

		const lines = this.items.map((item) => {
			const marker = { pending: "[ ]", in_progress: "[>]", completed: "[x]" }[
				item.status
			];
			return `${marker} #${item.id}: ${item.text}`;
		});

		const done = this.items.filter((t) => t.status === "completed").length;

		lines.push(`\n(Progress: ${done}/${this.items.length} completed)`);
		return lines.join("\n");
	}
}

const TODO = new TodoManager();

/**
 * 工具函数实现：供父子智能体共同调用
 */

/**
 * 安全路径校验：确保访问不超出工作目录范围
 */
function safePath(p: string): string {
	const resolvedPath = path.resolve(WORKDIR, p);
	if (!resolvedPath.startsWith(WORKDIR)) {
		throw new Error(`安全拦截：禁止访问工作区之外的路径 -> ${p}`);
	}
	return resolvedPath;
}

/**
 * 执行 PowerShell 命令
 * @param command 要执行的命令字符串
 */
function runPwsh(command: string): string {
	const dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
	if (dangerous.some((d) => command.includes(d))) {
		return "Error: Dangerous command blocked";
	}

	try {
		// 使用 execSync 模拟同步执行
		const output = execSync(command, {
			cwd: WORKDIR,
			timeout: 120000, // 超时时间 120s
			encoding: "utf-8",
			stdio: ["pipe", "pipe", "pipe"], // 捕获 stdout 和 stderr
			shell: "pwsh.exe",
		});
		return output.trim() || "(no output)";
	} catch (error: any) {
		if (error.code === "ETIMEDOUT") {
			return "Error: Timeout (120s)";
		}
		// 合并输出 stdout 和错误 stderr
		const out =
			(error.stdout?.toString() || "") + (error.stderr?.toString() || "");
		return out.trim().slice(0, 50000) || `Error: ${error.message}`;
	}
}

/**
 * 读取文件内容
 */
function runRead(filePath: string): string {
	try {
		const sp = safePath(filePath);
		return fs.readFileSync(sp, "utf-8");
	} catch (err: any) {
		return `Read error: ${err.message}`;
	}
}

/**
 * 写入（覆盖）文件内容
 */
function runWrite(filePath: string, content: string): string {
	try {
		const sp = safePath(filePath);
		// 自动递归创建目录
		fs.mkdirSync(path.dirname(sp), { recursive: true });
		fs.writeFileSync(sp, content, "utf-8");
		return `Write success, ${content.length} characters written to ${filePath}`;
	} catch (err: any) {
		return `Write error: ${err.message}`;
	}
}

/**
 * 编辑文件：精准替换文件中的特定文本
 */
function runEdit(filePath: string, oldText: string, newText: string): string {
	try {
		const sp = safePath(filePath);
		let content = fs.readFileSync(sp, "utf-8");
		if (!content.includes(oldText))
			return `Error: Cannot find text to replace "${oldText}"`;
		// 替换首个匹配项并写回
		content = content.replace(oldText, newText);
		fs.writeFileSync(sp, content, "utf-8");
		return `Edit success: ${filePath}`;
	} catch (err: any) {
		return `Edit error: ${err.message}`;
	}
}

/**
 * 工具分发器：将 OpenAI 的 tool_calls 映射到本地函数
 */
const TOOLS_HANDLERS: Record<string, (...args: any[]) => any> = {
	powershell: (args: any) => runPwsh(args.command),
	read_file: (args: any) => runRead(args.filePath),
	write_file: (args: any) => runWrite(args.filePath, args.content),
	edit_file: (args: any) => runEdit(args.filePath, args.oldText, args.newText),
	todo: (args: any) => TODO.update(args.items),
};

const CHILD_TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
	{
		type: "function",
		function: {
			name: "powershell",
			description: "Run a powershell command.",
			parameters: {
				type: "object",
				properties: { command: { type: "string" } },
				required: ["command"],
			},
		},
	},
	{
		type: "function",
		function: {
			name: "read_file",
			description: "Read a file.",
			parameters: {
				type: "object",
				properties: { filePath: { type: "string" } },
				required: ["filePath"],
			},
		},
	},
	{
		type: "function",
		function: {
			name: "write_file",
			description: "Write a file.",
			parameters: {
				type: "object",
				properties: {
					filePath: { type: "string" },
					content: { type: "string" },
				},
				required: ["filePath", "content"],
			},
		},
	},
	{
		type: "function",
		function: {
			name: "edit_file",
			description: "Edit a file.",
			parameters: {
				type: "object",
				properties: {
					filePath: { type: "string" },
					oldText: { type: "string" },
					newText: { type: "string" },
				},
				required: ["filePath", "oldText", "newText"],
			},
		},
	},
	{
		type: "function",
		function: {
			name: "todo",
			description: "Update task list. Track progress on multi-step tasks.",
			parameters: {
				type: "object",
				properties: {
					items: {
						type: "array",
						items: {
							type: "object",
							properties: {
								id: { type: "string" },
								text: { type: "string" },
								status: {
									type: "string",
									enum: ["pending", "in_progress", "completed"],
								},
							},
							required: ["id", "text", "status"],
						},
					},
				},
				required: ["items"],
			},
		},
	},
];

/**
 * 运行子智能体：拥有独立的上下文，用于处理特定的原子任务
 * @param prompt 任务的具体指令
 * @returns 任务执行完毕后的总结字符串
 */
async function runSubagent(prompt: string): Promise<string> {
	const sub_messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
		{
			role: "system",
			content: SUBAGENT_SYSTEM,
		},
		{ role: "user", content: prompt },
	];

	let lastMessage: OpenAI.Chat.ChatCompletionMessage | null = null;

	for (let i = 0; i < 30; i++) {
		const response = await openai.chat.completions.create({
			model: MODEL,
			messages: sub_messages,
			tools: CHILD_TOOLS,
			max_completion_tokens: 8000,
		});

		lastMessage = response.choices[0].message;
		sub_messages.push(lastMessage);

		if (
			response.choices[0].finish_reason !== "tool_calls" ||
			!lastMessage.tool_calls
		) {
			break;
		}

		for (const toolCall of lastMessage.tool_calls) {
			if (toolCall.type === "function") {
				const toolName = toolCall.function.name;
				const args = JSON.parse(toolCall.function.arguments);
				const handler = TOOLS_HANDLERS[toolName];
				if (!handler) {
					console.log(`Error: Unknown tool ${toolName}`);
					continue;
				}
				const result = handler(args);
				console.log(
					`\x1b[33m[子智能体调用工具]\x1b[0m -> ${toolName}: ${result.slice(0, 200)}...`,
				);
				sub_messages.push({
					role: "tool",
					tool_call_id: toolCall.id,
					content: String(result),
				});
			}
		}
	}

	return lastMessage?.content || "(no summary)";
}

const PARENT_TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
	...CHILD_TOOLS,
	{
		type: "function",
		function: {
			name: "task",
			description:
				"Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
			parameters: {
				type: "object",
				properties: {
					prompt: { type: "string" },
				},
				required: ["prompt"],
			},
		},
	},
];

/**
 * 智能体核心逻辑循环：处理对话、调用工具并进行状态监控
 * @param messages 历史对话消息队列
 */
async function agentLoop(messages: OpenAI.Chat.ChatCompletionMessageParam[]) {
	let rounds_since_todo = 0; // 记录距上一次更新 Todo 的对话轮数

	while (true) {
		const response = await openai.chat.completions.create({
			model: MODEL,
			messages: [{ role: "system", content: SYSTEM }, ...messages],
			tools: PARENT_TOOLS,
			max_completion_tokens: 8000,
		});

		const message = response.choices[0].message;

		// 将助理的回复（可能包含 tool_calls）存入历史
		messages.push(message);

		// 如果没有工具调用，结束循环
		if (
			response.choices[0].finish_reason !== "tool_calls" ||
			!message.tool_calls
		) {
			return;
		}

		let usedTodoThisRound = false;

		// 处理每一个工具调用
		for (const toolCall of message.tool_calls) {
			if (toolCall.type === "function") {
				const toolName = toolCall.function.name;
				const args = JSON.parse(toolCall.function.arguments);
				let result = "";

				if (toolName === "todo") usedTodoThisRound = true;
				if (toolName === "task") {
					result = await runSubagent(args.prompt);
				} else {
					const handler = TOOLS_HANDLERS[toolName];
					if (!handler) {
						console.log(`Error: Unknown tool ${toolName}`);
						continue;
					}
					result = handler(args);
					console.log(
						`\x1b[32m[父智能体调用工具]\x1b[0m -> ${toolName}: ${result.slice(0, 200)}...`,
					);
				}

				// OpenAI 要求将结果作为 role: "tool" 反馈
				messages.push({
					role: "tool",
					tool_call_id: toolCall.id,
					content: result,
				});
			}
		}

		if (usedTodoThisRound) {
			rounds_since_todo = 0;
			console.log(`\x1b[34m[Todo List]\n${TODO.render()}\x1b[0m`);
		} else {
			rounds_since_todo++;
		}
		if (rounds_since_todo > 3) {
			console.log(
				`\x1b[31m[系统监工] 大模型已连续 3 轮未更新状态，强制提醒！\x1b[0m`,
			);
			messages.push({
				role: "user",
				content: "<reminder>Update your todos.</reminder>",
			});
			rounds_since_todo = 0;
		}
	}
}

/**
 * CLI 交互主程序：负责终端输入输出、维护全局对话历史
 */
async function main() {
	const rl = readline.createInterface({
		input: process.stdin,
		output: process.stdout,
	});

	const ask = (query: string) =>
		new Promise<string>((resolve) => rl.question(query, resolve));
	const history: OpenAI.Chat.ChatCompletionMessageParam[] = [];

	while (true) {
		const query = await ask("\x1b[36mMy-agent >> \x1b[0m");

		if (!query || ["q", "exit"].includes(query.toLowerCase().trim())) {
			rl.close();
			break;
		}

		// 将用户输入加入历史
		history.push({ role: "user", content: query });
		// 进入智能体逻辑循环
		await agentLoop(history);

		// 打印最后一条非工具调用的助理回复文本
		const lastMsg = history[history.length - 1];
		if (lastMsg.role === "assistant" && lastMsg.content) {
			console.log(`\n${lastMsg.content}`);
		}
		console.log();
	}
}

main().catch(console.error);
