from __future__ import annotations

DEFAULT_TEXT_KEY = "Content"
DEFAULT_MIN_N = 1
DEFAULT_MAX_N = 10
DEFAULT_FLUSH_THRESHOLD = 200_000
DEFAULT_END_CHARS = "。！？"
DEFAULT_VERBOSITY = "normal"
DEFAULT_PROMPT_MODE = "random"
DEFAULT_N_SELECTION_MODE = "weighted"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 0
DEFAULT_TOP_P = 1.0
DEFAULT_N_TEMPERATURE = 1.0
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_REPETITION_WINDOW = 20

RANDOM_SEED_PROMPTS = [
	"你今天在想什么？",
	"你现在最想写的一句话是？",
	"如果现在必须写点什么，你会从哪里开始？",
	"今天有没有一个瞬间让你停下来思考？",
	"最近有什么一直在你脑海里打转？",
	"你现在的心情像什么？",
	"有没有一句话你一直想说却没说出口？",
	"如果可以对一个人说一句话，你会写什么？",
	"最近有没有什么让你感到矛盾或复杂的情绪？",
	"写下你此刻最真实的感受",
	"讲一个今天发生的小故事吧",
	"从“刚刚，我突然发现……”开始写",
	"描述一个你印象深刻的瞬间",
	"写一段你不想忘记的记忆",
	"如果今天是一篇文章，它的开头会是什么？",
	"如果一切可以重新开始，你会写什么？",
	"假设你身处另一个世界，第一句话是什么？",
	"给未来的自己写一句话",
	"如果今天有隐藏剧情，它会是什么？",
	"从“其实事情并没有看起来那么简单……”开始",
	"用一句话总结今天",
	"用三个词形容现在的你",
	"写下你此刻看到的画面",
	"接着这句话写下去：“我也不知道为什么，但是……”",
	"从“也许这并不重要，但我还是想说……”开始",
]
