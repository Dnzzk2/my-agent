/*

生成一个拥有独立 messages=[] 的子智能体。子智能体在其自身的上下文中工作，
共享文件系统，完成后仅向父智能体返回任务摘要。

    父智能体 (Parent)                  子智能体 (Subagent)
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- 全新上下文
    |                  |    派发     |                  |
    | 工具: task        | ---------->| while 工具调用:    |
    |   提示词="..."    |            |   执行工具动作     |
    |   描述=""        |            |   追加执行结果     |
    |                  |    摘要     |                  |
    |   结果 = "..."    | <--------- | 返回最终文本总结   |
    +------------------+             +------------------+
              |
    父级上下文保持整洁。
    子级上下文随后丢弃。

核心见解：“进程隔离免费带来了上下文隔离。”

*/

import { execSync } from "node:child_process";
import OpenAI from "openai";
import "dotenv/config";
import * as fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

// 全局配置：WORKDIR 为当前执行路径
const WORKDIR = process.cwd();
const MODEL = "deepseek-chat";
// 父级系统提示词：鼓励使用 task 工具进行分工
const SYSTEM = `You are a coding agent at ${WORKDIR}. Use the task tool to delegate exploration or subtasks.`;
// 子级系统提示词：要求完成任务并总结
const SUBAGENT_SYSTEM = `You are a coding subagent at ${WORKDIR}. Complete the given task, then summarize your findings.`;

const openai = new OpenAI({
	baseURL: process.env.DEEPSEEK_BASE_URL,
	apiKey: process.env.DEEPSEEK_API_KEY,
});

/**
 * 安全路径：限制文件操作仅能在当前工作区内进行
 */
function safePath(p: string): string {
	const resolvedPath = path.resolve(WORKDIR, p);
	if (!resolvedPath.startsWith(WORKDIR)) {
		throw new Error(`安全拦截：禁止访问工作区之外的路径 -> ${p}`);
	}
	return resolvedPath;
}

/**
 * 执行系统命令工具
 */
function runPwsh(command: string): string {
	const dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
	if (dangerous.some((d) => command.includes(d))) {
		return "Error: 危险命令已被拒绝";
	}

	try {
		const output = execSync(command, {
			cwd: WORKDIR,
			timeout: 120000,
			encoding: "utf-8",
			stdio: ["pipe", "pipe", "pipe"],
			shell: "pwsh.exe",
		});
		return output.trim() || "(无输出)";
	} catch (error: any) {
		if (error.code === "ETIMEDOUT") {
			return "Error: 执行超时 (120s)";
		}
		const out = (error.stdout?.toString() || "") + (error.stderr?.toString() || "");
		return out.trim().slice(0, 50000) || `Error: ${error.message}`;
	}
}

function runRead(filePath: string): string {
	try {
		const sp = safePath(filePath);
		return fs.readFileSync(sp, "utf-8");
	} catch (err: any) {
		return `读取错误: ${err.message}`;
	}
}

function runWrite(filePath: string, content: string): string {
	try {
		const sp = safePath(filePath);
		fs.mkdirSync(path.dirname(sp), { recursive: true });
		fs.writeFileSync(sp, content, "utf-8");
		return `写入成功，长度: ${content.length}`;
	} catch (err: any) {
		return `写入错误: ${err.message}`;
	}
}

function runEdit(filePath: string, oldText: string, newText: string): string {
	try {
		const sp = safePath(filePath);
		let content = fs.readFileSync(sp, "utf-8");
		if (!content.includes(oldText))
			return `错误：找不到旧文本 "${oldText}"`;
		content = content.replace(oldText, newText);
		fs.writeFileSync(sp, content, "utf-8");
		return `编辑成功: ${filePath}`;
	} catch (err: any) {
		return `编辑错误: ${err.message}`;
	}
}

// 通用的工具处理器映射
const TOOLS_HANDLERS: Record<string, (...args: any[]) => any> = {
	powershell: (args: any) => runPwsh(args.command),
	read_file: (args: any) => runRead(args.filePath),
	write_file: (args: any) => runWrite(args.filePath, args.content),
	edit_file: (args: any) => runEdit(args.filePath, args.oldText, args.newText),
};

// 子智能体可用的工具定义（不包含 task，防止无限套娃）
const CHILD_TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
	{
		type: "function",
		function: {
			name: "powershell",
			description: "运行 PowerShell 命令。",
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
			description: "读取一个文件。",
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
			description: "写入文件内容。",
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
			description: "编辑现有文件。",
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
];

/**
 * 运行子智能体逻辑：
 * 拥有全新的消息队列，独立于父级的对话历史。
 */
async function runSubagent(prompt: string): Promise<string> {
	// 子智能体的初始状态：仅包含系统提示词和分配的任务
	const sub_messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
		{ role: "system", content: SUBAGENT_SYSTEM },
		{ role: "user", content: prompt },
	];

	let lastMessage: OpenAI.Chat.ChatCompletionMessage | null = null;

	// 限制子智能体对话次数（此处最大 30 次）以防死循环
	for (let i = 0; i < 30; i++) {
		const response = await openai.chat.completions.create({
			model: MODEL,
			messages: sub_messages,
			tools: CHILD_TOOLS,
			max_completion_tokens: 8000,
		});

		lastMessage = response.choices[0].message;
		sub_messages.push(lastMessage);

		// 如果不再需要工具调用，表示任务完成
		if (response.choices[0].finish_reason !== "tool_calls" || !lastMessage.tool_calls) {
			break;
		}

		// 处理子智能体的工具请求
		for (const toolCall of lastMessage.tool_calls) {
			if (toolCall.type === "function") {
				const toolName = toolCall.function.name;
				const args = JSON.parse(toolCall.function.arguments);
				const handler = TOOLS_HANDLERS[toolName];
				if (!handler) {
					console.log(`Error: 找不到工具 ${toolName}`);
					continue;
				}
				const result = handler(args);
				
				// 打印子级工作日志
				console.log(`\x1b[33m[子智能体调用工具]\x1b[0m -> ${toolName}: ${result.slice(0, 200)}...`);

				sub_messages.push({
					role: "tool",
					tool_call_id: toolCall.id,
					content: String(result),
				});
			}
		}
	}

	// 最终仅向上层返回子智能体最后给出的结果文字（即任务摘要）
	return lastMessage?.content || "(无总结内容)";
}

// 父智能体的工具列表：包含基础工具及 task (生成子任务)
const PARENT_TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
	...CHILD_TOOLS,
	{
		type: "function",
		function: {
			name: "task",
			description: "生成一个子智能体并进入独立的运行上下文，主进程在此期间会等待子进程反馈。",
			parameters: {
				type: "object",
				properties: { prompt: { type: "string" } },
				required: ["prompt"],
			},
		},
	},
];

/**
 * 主循环逻辑
 */
async function agentLoop(messages: OpenAI.Chat.ChatCompletionMessageParam[]) {
	while (true) {
		const response = await openai.chat.completions.create({
			model: MODEL,
			messages: [{ role: "system", content: SYSTEM }, ...messages],
			tools: PARENT_TOOLS,
			max_completion_tokens: 8000,
		});

		const message = response.choices[0].message;
		messages.push(message);

		// 结束对话
		if (response.choices[0].finish_reason !== "tool_calls" || !message.tool_calls) {
			return;
		}

		for (const toolCall of message.tool_calls) {
			if (toolCall.type === "function") {
				const toolName = toolCall.function.name;
				const args = JSON.parse(toolCall.function.arguments);
				let result = "";

				// 判读是否为特殊工具：生成子代理
				if (toolName === "task") {
					result = await runSubagent(args.prompt);
				} else {
					const handler = TOOLS_HANDLERS[toolName];
					if (!handler) {
						console.log(`Error: 找不到工具 ${toolName}`);
						continue;
					}
					result = handler(args);
					
					// 打印父级工作日志
					console.log(`\x1b[32m[父智能体调用工具]\x1b[0m -> ${toolName}: ${result.slice(0, 200)}...`);
				}

				// 将（子级产生的摘要或常规工具执行结果）反馈回全局上下文
				messages.push({
					role: "tool",
					tool_call_id: toolCall.id,
					content: result,
				});
			}
		}
	}
}

/**
 * 入口函数
 */
async function main() {
	const rl = readline.createInterface({
		input: process.stdin,
		output: process.stdout,
	});

	const ask = (query: string) => new Promise<string>((resolve) => rl.question(query, resolve));
	const history: OpenAI.Chat.ChatCompletionMessageParam[] = [];

	while (true) {
		const query = await ask("\x1b[36mMy-agent >> \x1b[0m");

		if (!query || ["q", "exit"].includes(query.toLowerCase().trim())) {
			rl.close();
			break;
		}

		history.push({ role: "user", content: query });
		await agentLoop(history);

		const lastMsg = history[history.length - 1];
		if (lastMsg.role === "assistant" && lastMsg.content) {
			console.log(`\n${lastMsg.content}`);
		}
		console.log();
	}
}

main().catch(console.error);
