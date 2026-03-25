import { execSync } from "node:child_process";
import OpenAI from "openai";
import "dotenv/config";
import * as fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

// 全局配置
const WORKDIR = process.cwd();
const SKILLS_DIR = path.join(WORKDIR, "skills");
const MODEL = "deepseek-chat";

const openai = new OpenAI({
	baseURL: process.env.DEEPSEEK_BASE_URL,
	apiKey: process.env.DEEPSEEK_API_KEY,
});

/**
 * 技能加载器
 */

class SkillLoader {
	private skills: Record<string, { desc: string; body: string }> = {};

	constructor() {
		this.loadAll();
	}

	private findSkillFiles(dir: string): string[] {
		const entries = fs.readdirSync(dir, { withFileTypes: true });
		const files: string[] = [];

		for (const entry of entries) {
			const fullPath = path.join(dir, entry.name);

			if (entry.isDirectory()) {
				files.push(...this.findSkillFiles(fullPath));
				continue;
			}

			if (entry.isFile() && entry.name.endsWith(".md")) {
				files.push(fullPath);
			}
		}

		return files;
	}

	// 扫描 skills 文件夹下所有的 .md 文件
	loadAll() {
		if (!fs.existsSync(SKILLS_DIR)) return;
		const files = this.findSkillFiles(SKILLS_DIR);
		for (const filePath of files) {
			const content = fs.readFileSync(filePath, "utf-8");
			const fileName = path.basename(filePath);
			const defaultName =
				fileName.toUpperCase() === "SKILL.MD"
					? path.basename(path.dirname(filePath))
					: path.basename(filePath, ".md");

			// 简单解析 YAML frontmatter (由 --- 包围的部分)
			const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)/);
			if (match) {
				const metaBlock = match[1];
				const body = match[2].trim();

				let name = defaultName;
				let desc = "无描述";

				metaBlock.split("\n").forEach((line) => {
					if (line.startsWith("name:")) name = line.split(":")[1].trim();
					if (line.startsWith("description:")) desc = line.split(":")[1].trim();
				});

				this.skills[name] = { desc, body };
			}
		}
	}

	// 第一层：生成目录大纲，给LLM看,有哪些技能，塞入系统提示词
	getDescriptions(): string {
		const names = Object.keys(this.skills);
		if (names.length === 0) return "(当前无可用技能)";
		return names.map((name) => `${name}: ${this.skills[name].desc}`).join("\n");
	}

	// 第二层：根据用户意图，从技能库中筛选出最相关的技能，塞入用户提示词
	getContent(name: string): string {
		const skill = this.skills[name];
		if (!skill)
			return `错误：找不到名为 '${name}' 的技能。可用技能有: ${Object.keys(
				this.skills,
			).join(", ")}`;
		return `<skill name="${name}">\n${skill.body}\n</skill>`;
	}
}

const skillLoader = new SkillLoader();

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
	load_skill: (args: any) => skillLoader.getContent(args.name),
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
			name: "load_skill",
			description: "按需加载专业领域的详细知识或最佳实践。",
			parameters: {
				type: "object",
				properties: {
					name: { type: "string", description: "要加载的技能名称" },
				},
				required: ["name"],
			},
		},
	},
];

console.log(skillLoader.getDescriptions());

const SYSTEM = `You are a coding agent at ${WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.
Skills available:
${skillLoader.getDescriptions()}`;

/**
 * 智能体主逻辑循环
 */
async function agentLoop(messages: OpenAI.Chat.ChatCompletionMessageParam[]) {
	while (true) {
		const response = await openai.chat.completions.create({
			model: MODEL,
			messages: [{ role: "system", content: SYSTEM }, ...messages],
			tools: TOOLS,
			max_completion_tokens: 8000,
		});

		const message = response.choices[0].message;
		messages.push(message);

		console.log(message);

		if (
			response.choices[0].finish_reason !== "tool_calls" ||
			!message.tool_calls
		) {
			return;
		}

		// 依次处理模型请求的工具调用
		for (const toolCall of message.tool_calls) {
			if (toolCall.type === "function") {
				const toolName = toolCall.function.name;
				const args = JSON.parse(toolCall.function.arguments);

				const handler = TOOLS_HANDLERS[toolName];
				if (!handler) {
					console.log(`Error: 找不到工具 ${toolName}`);
					continue;
				}
				const result = handler(args);

				// 打印日志：保留缩进且截取关键内容显示
				console.log(
					`\x1b[32m[智能体调用工具]\x1b[0m -> ${toolName}: ${result.slice(0, 200)}...`,
				);

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
 * 程序入口：负责 CLI 交互
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
