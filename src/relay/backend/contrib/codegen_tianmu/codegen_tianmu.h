#ifndef TVM_RELAY_BACKEND_CONTRIB_CODEGEN_TIANMU_CODEGEN_TIANMU_H_
#define TVM_RELAY_BACKEND_CONTRIB_CODEGEN_TIANMU_CODEGEN_TIANMU_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/container.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {

struct Output {
    std::string name;
    std::string dtype;
    int size;
    bool need_copy;
    Output() {
        name = "";
        dtype = "";
        size = 0;
        need_copy = true;
    }
};

class TianMuSourceModuleCodegenBase {
public:
    TianMuSourceModuleCodegenBase() = default;
    virtual ~TianMuSourceModuleCodegenBase() = default;

    virtual runtime::Module CreateTianMuSourceModule(const ObjectRef& ref) = 0;
};

class CodegenTianMuBase {
public:
    virtual ~CodegenTianMuBase() {}
protected:
    // virtual std::string JIT(const std::vector<Output>& out) = 0;
    std::ostringstream code_stream_;
private:
    int indent_{0};
};

}
}
}

#endif