/*
模型通过 TodoManager 追踪自身进度。
当模型忘记更新状态时，系统会通过强制提醒（Nag Reminder）来促使其更新。

    +----------+      +-------+      +---------+
    |   用户   | ---> |  LLM  | ---> | 工具     |
    |   提示词 |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   工具执行结果 |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager 状态      |
                    | [ ] 任务 A            |
                    | [>] 任务 B <- 执行中  |
                    | [x] 任务 C            |
                    +-----------------------+
                                |
                    如果 连续 3 轮未更新:
                      注入 <reminder> 强制提醒

核心见解：“智能体可以追踪规划自身进度，且主进程全局可见。”
*/

import { execSync } from "node:child_process";
import OpenAI from "openai";
import "dotenv/config";
import * as fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

// 全局配置
const WORKDIR = process.cwd();
const MODEL = "deepseek-chat";
const SYSTEM = `You are a coding agent at ${WORKDIR}. Use bash to solve tasks. Act, don't explain.`;

const openai = new OpenAI({
	baseURL: process.env.DEEPSEEK_BASE_URL,
	apiKey: process.env.DEEPSEEK_API_KEY,
});

/**
 * 任务管理器：负责维护 Todo 列表的状态
 */
class TodoManager {
	items: { id: string; text: string; status: string }[] = [];

	// 更新任务列表
	update(newItems: any[]): string {
		if (newItems.length > 20) throw new Error("最多只允许 20 个任务项");

		let inProgressCount = 0;
		const validated = [];

		for (let i = 0; i < newItems.length; i++) {
			const item = newItems[i];
			const text = String(item.text || "").trim();
			const status = String(item.status || "pending").toLowerCase();
			const id = String(item.id || String(i + 1));

			if (!text) throw new Error(`任务 ${id}：内容不能为空`);
			if (!["pending", "in_progress", "completed"].includes(status))
				throw new Error(`任务 ${id}：无效的状态 ${status}`);

			if (status === "in_progress") inProgressCount++;

			validated.push({ id, text, status });
		}

		// 约束：同一时间只能有一个任务处于“进行中”
		if (inProgressCount > 1)
			throw new Error("同一时间只能有一个任务处于 'in_progress' 状态");

		this.items = validated;
		return this.render();
	}

	// 将任务列表渲染成可读字符串，供控制台展示和发送给模型
	render(): string {
		if (this.items.length === 0) return "当前无任务项。";

		const lines = this.items.map((item) => {
			const marker = { pending: "[ ]", in_progress: "[>]", completed: "[x]" }[
				item.status
			];
			return `${marker} #${item.id}: ${item.text}`;
		});

		const done = this.items.filter((t) => t.status === "completed").length;
		lines.push(`\n(进度：已完成 ${done} / 总计 ${this.items.length})`);
		return lines.join("\n");
	}
}

const TODO = new TodoManager();

/**
 * 安全路径校验：防止模型通过 ../ 访问工作区以外的文件
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
 */
function runPwsh(command: string): string {
	const dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
	if (dangerous.some((d) => command.includes(d))) {
		return "Error: 危险命令已被拦截";
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

/**
 * 读取文件内容
 */
function runRead(filePath: string): string {
	try {
		const sp = safePath(filePath);
		return fs.readFileSync(sp, "utf-8");
	} catch (err: any) {
		return `读取失败: ${err.message}`;
	}
}

/**
 * 完整写入文件
 */
function runWrite(filePath: string, content: string): string {
	try {
		const sp = safePath(filePath);
		fs.mkdirSync(path.dirname(sp), { recursive: true });
		fs.writeFileSync(sp, content, "utf-8");
		return `写入成功，共 ${content.length} 字符`;
	} catch (err: any) {
		return `写入失败: ${err.message}`;
	}
}

/**
 * 编辑文件（局部替换）
 */
function runEdit(filePath: string, oldText: string, newText: string): string {
	try {
		const sp = safePath(filePath);
		let content = fs.readFileSync(sp, "utf-8");
		if (!content.includes(oldText))
			return `错误：在文件中找不到要替换的内容 "${oldText}"`;
		content = content.replace(oldText, newText);
		fs.writeFileSync(sp, content, "utf-8");
		return `编辑成功: ${filePath}`;
	} catch (err: any) {
		return `编辑失败: ${err.message}`;
	}
}

// 映射工具名称到具体函数
const TOOLS_HANDLERS: Record<string, (...args: any[]) => any> = {
	powershell: (args: any) => runPwsh(args.command),
	read_file: (args: any) => runRead(args.filePath),
	write_file: (args: any) => runWrite(args.filePath, args.content),
	edit_file: (args: any) => runEdit(args.filePath, args.oldText, args.newText),
	todo: (args: any) => TODO.update(args.items),
};

// 定义发给大模型的工具列表
const TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
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
			description: "读取文件内容。",
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
			description: "将内容写入文件。",
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
			description: "替换文件中的特定文本内容。",
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
			description: "更新任务列表。通过它来拆解多步子任务并追踪进度。",
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
 * 智能体主逻辑循环
 */
async function agentLoop(messages: OpenAI.Chat.ChatCompletionMessageParam[]) {
	let rounds_since_todo = 0; // 记录距上一次更新状态经过了多少轮对话

	while (true) {
		const response = await openai.chat.completions.create({
			model: MODEL,
			messages: [{ role: "system", content: SYSTEM }, ...messages],
			tools: TOOLS,
			max_completion_tokens: 8000,
		});

		const message = response.choices[0].message;
		messages.push(message);

		if (response.choices[0].finish_reason !== "tool_calls" || !message.tool_calls) {
			return;
		}

		let usedTodoThisRound = false;

		// 依次处理模型请求的工具调用
		for (const toolCall of message.tool_calls) {
			if (toolCall.type === "function") {
				const toolName = toolCall.function.name;
				const args = JSON.parse(toolCall.function.arguments);

				if (toolName === "todo") usedTodoThisRound = true;

				const handler = TOOLS_HANDLERS[toolName];
				if (!handler) {
					console.log(`Error: 找不到工具 ${toolName}`);
					continue;
				}
				const result = handler(args);

				// 打印日志：保留缩进且截取关键内容显示
				console.log(`\x1b[32m[智能体调用工具]\x1b[0m -> ${toolName}: ${result.slice(0, 200)}...`);

				messages.push({
					role: "tool",
					tool_call_id: toolCall.id,
					content: result,
				});
			}
		}

		// 状态监控逻辑
		if (usedTodoThisRound) {
			rounds_since_todo = 0;
			console.log(`\x1b[34m[Todo List]\n${TODO.render()}\x1b[0m`);
		} else {
			rounds_since_todo++;
		}

		// 如果大模型连续 3 轮都没通过 todo 更新状态，强制插入提醒
		if (rounds_since_todo > 3) {
			console.log(`\x1b[31m[系统监工] 大模型已连续 3 轮未更新状态，强制提醒更新 Todo！\x1b[0m`);
			messages.push({
				role: "user",
				content: "<reminder>请更新你的 todo 列表以同步任务进度。</reminder>",
			});
			rounds_since_todo = 0;
		}
	}
}

/**
 * 程序入口：负责 CLI 交互
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
