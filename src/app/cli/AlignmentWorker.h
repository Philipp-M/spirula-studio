#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace spirula::align {
// Engine and inference contexts stay on one thread while photo workers fit cameras.
class GpuWorker {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::function<void()>> tasks;
    bool stopping=false;
    std::thread worker{[this] {
        for(;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex);
                ready.wait(lock,[&]{return stopping||!tasks.empty();});
                if(tasks.empty())return;
                task=std::move(tasks.front());tasks.pop_front();
            }
            task();
        }
    }};
public:
    ~GpuWorker() {
        {std::lock_guard lock(mutex);stopping=true;}
        ready.notify_one();worker.join();
    }
    template<class F> auto run(F&& f) {
        using Result=std::invoke_result_t<F>;
        auto task=std::make_shared<std::packaged_task<Result()>>(std::forward<F>(f));
        auto result=task->get_future();
        {std::lock_guard lock(mutex);tasks.emplace_back([task]{(*task)();});}
        ready.notify_one();return result.get();
    }
};
}
