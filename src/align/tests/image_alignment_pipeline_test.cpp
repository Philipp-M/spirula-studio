#include "app/cli/AlignmentWorker.h"
#include "sfm/geometry/AbsolutePose.h"
#include <iostream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    try {
        spirula::align::GpuWorker gpu;
        const auto thread=gpu.run([]{return std::this_thread::get_id();});
        std::promise<void> cpu_ready,continue_cpu;
        auto resume=continue_cpu.get_future();
        auto first=std::async(std::launch::async,[&] {
            if(gpu.run([]{return std::this_thread::get_id();})!=thread)throw std::runtime_error("GPU thread changed");
            cpu_ready.set_value();resume.get();
            return gpu.run([]{return 1;});
        });
        cpu_ready.get_future().get();
        const auto second=gpu.run([]{return 2;});
        continue_cpu.set_value();
        if(first.get()!=1||second!=2)throw std::runtime_error("CPU camera work blocked the next GPU view");
        bool caught=false;
        try {gpu.run([]{throw std::runtime_error("test failure");});}catch(const std::runtime_error&){caught=true;}
        if(!caught||gpu.run([]{return 3;})!=3)throw std::runtime_error("GPU failure poisoned the queue");
        for(int dof:{3,6,7}) {
            auto residual=[](const sfm::Pose& pose,double scale,int j,double* r) {
                if(j%17==0){r[0]=r[1]=1e3;return;}
                const sfm::Vec3 point{std::sin(j*.71),std::cos(j*.53),4.+.01*j};
                const auto p=sfm::mul(pose.R,point)+pose.t;
                const auto q=sfm::mul(sfm::angleAxisToRotation({.01,-.02,.03}),point)+sfm::Vec3{.1,-.05,.02};
                r[0]=500*(scale*p.x/p.z-q.x/q.z);r[1]=500*(scale*p.y/p.z-q.y/q.z);
            };
            sfm::Pose reference;double reference_scale=1;
            sfm::pose_detail::lmRefine(300,residual,dof,reference,reference_scale,20);
            for(int threads:{1,2,4}) {
#ifdef _OPENMP
                omp_set_num_threads(threads);
#endif
                sfm::Pose parallel;double scale=1;
                sfm::pose_detail::lmRefine(300,residual,dof,parallel,scale,20,true);
                if(parallel.R!=reference.R||parallel.t.x!=reference.t.x||parallel.t.y!=reference.t.y||
                   parallel.t.z!=reference.t.z||scale!=reference_scale)
                    throw std::runtime_error("parallel pose evaluation changed ordered numerical results");
            }
        }
        std::cout<<"PASS: single GPU owner, cross-view CPU overlap and exception recovery\n";
        std::cout<<"PASS: serial/parallel LM equality for 3, 6 and 7 parameters and 1, 2 and 4 threads\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
