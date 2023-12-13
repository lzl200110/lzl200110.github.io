//! base/log.h
#ifndef BASE_LOG_HPP
#define BASE_LOG_HPP

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <format>
#include <iostream>
#include <mutex>
#include <ostream>
#include <source_location>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <vector>

//! Define log levels
#define LOG_LEVEL_FATAL 0  //!< Program will crash
#define LOG_LEVEL_ERROR 1  //!< Got serious problem, program can't handle
#define LOG_LEVEL_WARN 2   //!< Got inner abnormal situation, but program can handle it
#define LOG_LEVEL_NOTICE 3 //!< It not big problem, but we should notice it, such as invalid data input
#define LOG_LEVEL_INFO 4   //!< Normal message exchange with other program
#define LOG_LEVEL_DEBUG 5  //!< Normal process inside program
#define LOG_LEVEL_TRACE 6  //!< Temporary debugging log

//! Module ID
#ifndef LOG_MODULE_ID
#warning "Please define LOG_MODULE_ID as your module name, otherwise it will be NULL"
#define LOG_MODULE_ID NULL
#endif // LOG_MODULE_ID

//! Define commonly macros
#define LogPrintf(level, fmt, ...) \
	LogPrintfFunc(LOG_MODULE_ID, level, std::source_location::current(), fmt, ##__VA_ARGS__)

#define LogFatal(fmt, ...)  LogPrintf(LOG_LEVEL_FATAL, fmt, ##__VA_ARGS__)
#define LogErr(fmt, ...)    LogPrintf(LOG_LEVEL_ERROR, fmt, ##__VA_ARGS__)
#define LogWarn(fmt, ...)   LogPrintf(LOG_LEVEL_WARN, fmt, ##__VA_ARGS__)
#define LogNotice(fmt, ...) LogPrintf(LOG_LEVEL_NOTICE, fmt, ##__VA_ARGS__)
#define LogInfo(fmt, ...)   LogPrintf(LOG_LEVEL_INFO, fmt, ##__VA_ARGS__)
#define LogDbg(fmt, ...)    LogPrintf(LOG_LEVEL_DEBUG, fmt, ##__VA_ARGS__)
#define LogTrace(fmt, ...)  LogPrintf(LOG_LEVEL_TRACE, fmt, ##__VA_ARGS__)
#define LogTag() LogPrintf(LOG_LEVEL_TRACE, "==> Run Here <==")
#define LogUndo() LogPrintf(LOG_LEVEL_NOTICE, "!!! Undo !!!")

//! 打印错误码，需要 #include <string.h>
#define LogErrno(err, fmt, ...) LogErr("Errno:{}({}) " fmt, (err), strerror(err), ##__VA_ARGS__)

struct LogContent {
	const std::thread::id thread_id; //!< 线程ID
	const std::string timestamp;
	const std::string module_id; //!< 模块名
	const std::source_location location;
	const int level; //!< 日志等级
	const std::string text;
};

//! 定义日志输出函数
using LogPrintfFuncType = void (*)(const LogContent* content, void* ptr);

namespace {
	constexpr uint32_t LOG_MAX_LEN = (100 << 10);
	std::mutex _mtx;
	uint32_t _id_alloc = 0;
	struct OutputChannel {
		uint32_t id;
		LogPrintfFuncType func;
		void* ptr;
	};
	std::vector<OutputChannel> _output_channels;

	/**
	 * @brief 判断是否需要dispatch，_output_channels不为空则需要;
	 *
	 * @return true
	 * @return false
	 */
	bool can_dispatch() {
		std::unique_lock<std::mutex> _lock(_mtx);
		return !_output_channels.empty();
	}
	/**
	 * @brief
	 * 将content分配到注册的output_channels中，并调用output_channels中的回调函数
	 *
	 * @param content
	 */
	void dispatch(const LogContent& content) {
		std::unique_lock<std::mutex> _lock(_mtx);
		for (const auto& item : _output_channels) {
			if (item.func) {
				item.func(&content, item.ptr);
			}
		}
	}
} // namespace

/**
 * @brief  日志格式化打印实现
 * @param  module_id   Module Id
 * @param  location    location
 * @param  level       Log level
 * @param  fmt         Log format string
 */
template <typename... Args>
inline void LogPrintfFunc(const char* module_id, int level, const std::source_location location, const char* fmt,
                          Args... args) {
	if (!can_dispatch()) {
		return;
	}
	if (level < 0) {
		level = 0;
	}
	if (level > LOG_LEVEL_TRACE) {
		level = LOG_LEVEL_TRACE;
	}
	std::string module_id_print = (module_id != nullptr) ? module_id : "???";
	const std::chrono::zoned_time now{std::chrono::current_zone(), std::chrono::high_resolution_clock::now()};
	std::string text{};
	try {
		if (fmt != nullptr) {
			text = std::vformat(fmt, std::make_format_args(args...));
		}
	} catch (std::format_error& e) {
		LogPrintfFunc(module_id, LOG_LEVEL_WARN, location, "[format parse error]", e.what());
		return;
	}
	LogContent content = {.thread_id = std::this_thread::get_id(),
	                      .timestamp = std::format("{}", now),
	                      .module_id = std::move(module_id_print),
	                      .location = location,
	                      .level = level,
	                      .text = std::move(text)};

	dispatch(content);
}
/**
 * @brief 注册一个回调事件
 *
 * @param func
 * @param ptr
 * @return uint32_t
 */
inline uint32_t log_add_print_func(LogPrintfFuncType func, void* ptr) {
	std::unique_lock<std::mutex> _lock(_mtx);
	uint32_t new_id = ++_id_alloc;
	OutputChannel channel = {.id = new_id, .func = func, .ptr = ptr};
	_output_channels.push_back(channel);
	return new_id;
}
/**
 * @brief 移除一个回调事件
 *
 * @param id
 * @return true
 * @return false
 */
inline bool log_remove_print_func(uint32_t id) {
	std::unique_lock<std::mutex> _lock(_mtx);
	auto iter = std::remove_if(_output_channels.begin(), _output_channels.end(),
	                           [id](const OutputChannel& item) { return (item.id == id); });

	if (iter != _output_channels.end()) {
		_output_channels.erase(iter, _output_channels.end());
		return true;
	}
	return false;
}
namespace {
	const char* level_name = "FEWNIDT";
	const int level_color_num[] = {31, 91, 93, 33, 32, 36, 35};
	void _print_log_to_stdout(const LogContent* content) {
		{
			std::stringstream ss{};
			//! 开启色彩，显示日志等级
			ss << std::vformat("\033[{}m{} ",
			                   std::make_format_args(level_color_num[content->level], level_name[content->level]));
			//! 打印时间戳、线程号、模块名
			ss << std::format("{} ", content->timestamp);
			ss << content->thread_id;
			ss << std::format(" {} ", content->module_id);
			ss << content->location.function_name() << " ";
			ss << content->text;
			// ！ 打印函数名、文件名
			if (content->location.function_name() != nullptr)
				ss << std::format(" -- {}:{}", content->location.file_name(), content->location.line());
			//! 恢复色彩
			ss << std::string{"\033[0m"};
			std::cout << ss.str() << std::endl;
		}
	}
} // namespace

// ！ 你可以声明自己的log_output_filterfunc 来过滤 LogContent
bool __attribute((weak)) log_output_filter(const LogContent* content);

static void log_output_filterfunc(const LogContent* content, void* ptr);
static uint32_t id = 0;

/**
 * @brief 开启LogOutput
 *
 */
inline void LogOutput_Enable() {
	if (id == 0)
		id = log_add_print_func(log_output_filterfunc, nullptr);
}
/**
 * @brief 关闭LogOutput
 *
 */
inline void LogOutput_Disable() {
	log_remove_print_func(id);
	id = 0;
}

static void log_output_filterfunc(const LogContent* content, void*) {
	// log_output_filterfunc 不为nullptr
	// 就执行，返回true继续执行_print_log_to_stdout
	if ((log_output_filter != nullptr) && !log_output_filter(content))
		return;

	_print_log_to_stdout(content);
}
#endif
