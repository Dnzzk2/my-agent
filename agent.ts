/*
    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                         (loop continues)

*/

import OpenAI from "openai";
import { execSync } from "child_process";
import "dotenv/config";
import readline from "readline";

const openai = new OpenAI({
	baseURL: process.env.DEEPSEEK_BASE_URL,
	apiKey: process.env.DEEPSEEK_API_KEY,
});

const MODEL = "deepseek-chat";
const SYSTEM = `You are a coding agent at ${process.cwd()}. Use bash to solve tasks. Act, don't explain.`;

const TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
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
];

function runPwsh(command: string): string {
	const dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
	if (dangerous.some((d) => command.includes(d))) {
		return "Error: Dangerous command blocked";
	}

	try {
		// 使用 execSync 模拟同步执行
		const output = execSync(command, {
			cwd: process.cwd(),
			timeout: 120000, // 120s
			encoding: "utf-8",
			stdio: ["pipe", "pipe", "pipe"], // 捕获 stdout 和 stderr
			shell: "pwsh.exe",
		});
		return output.trim() || "(no output)";
	} catch (error: any) {
		if (error.code === "ETIMEDOUT") {
			return "Error: Timeout (120s)";
		}
		// 合并 stdout 和 stderr
		const out =
			(error.stdout?.toString() || "") + (error.stderr?.toString() || "");
		return out.trim().slice(0, 50000) || `Error: ${error.message}`;
	}
}

async function agentLoop(messages: OpenAI.Chat.ChatCompletionMessageParam[]) {
	while (true) {
		const response = await openai.chat.completions.create({
			model: MODEL,
			messages: [{ role: "system", content: SYSTEM }, ...messages],
			tools: TOOLS,
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

		// 处理每一个工具调用
		for (const toolCall of message.tool_calls) {
			if (
				toolCall.type === "function" &&
				toolCall.function.name === "powershell"
			) {
				const args = JSON.parse(toolCall.function.arguments);
				const command = args.command;

				console.log(`\x1b[33m$ ${command}\x1b[0m`);
				const output = runPwsh(command);
				console.log(output.slice(0, 200));

				// OpenAI 要求将结果作为 role: "tool" 反馈
				messages.push({
					role: "tool",
					tool_call_id: toolCall.id,
					content: output,
				});
			}
		}
	}
}

/**
 * CLI 交互主程序
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
		const query = await ask("\x1b[36ms01 >> \x1b[0m");

		if (!query || ["q", "exit"].includes(query.toLowerCase().trim())) {
			rl.close();
			break;
		}

		history.push({ role: "user", content: query });
		await agentLoop(history);

		// 打印最后一条非工具调用的文本回复
		const lastMsg = history[history.length - 1];
		if (lastMsg.role === "assistant" && lastMsg.content) {
			console.log(`\n${lastMsg.content}`);
		}
		console.log();
	}
}

main().catch(console.error);
