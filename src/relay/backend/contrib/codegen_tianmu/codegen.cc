#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <sstream>

#include <map>
#include <algorithm>

#include "../../utils.h"
#include "codegen_tianmu.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

class CodegenTianMu : public ExprVisitor, public CodegenTianMuBase {
public:
    std::ostringstream op_decl_stream;
    std::ostringstream mem_alloc_stream;
    explicit CodegenTianMu(const std::string& id) {
        this->ext_func_id_ = id; 
        op_decl_stream.flush();
    }
    
    int getTensorIdx() const {
        return tensor_idx;
    }

    void VisitExprDefault_(const Object* op) final {

    }
    
    void VisitExpr_(const VarNode* node) final {
        // std::cout << "TianMu CodeGen VisitExpr_(VarNode)" << std::endl;
        // Output output;
        // return {output};
        std::cout << node->name_hint() << std::endl;
    }

    void VisitExpr_(const GlobalVarNode* node) final {
        std::cout << node->name_hint << std::endl;
    }

    void VisitExpr_(const TupleNode* node) final {
        for (auto field : node->fields) {
            VisitExpr(field);
        }
    }

    void VisitExpr_(const TupleGetItemNode* op) final {
        VisitExpr(op->tuple);
    }

    void VisitExpr_(const ConstantNode* cn) final {

    }

    void VisitExpr_(const CallNode* call) final {
        auto op_node = call->op.as<OpNode>();
        std::cout << op_node->name << " " << op_node->op_type << std::endl;
        // std::cout << op_node->name << " " << op_node->num_inputs << std::endl;
        op_decl_stream << op_node->name << "(";
        for (size_t i = 0; i < call->args.size(); i++) {
            if (const auto* vn = call->args[i].as<VarNode>()) {
                op_decl_stream << vn->name_hint() << ", ";
            }
        }
        if (const auto* attrs_node = call->attrs.as<BaseAttrsNode>()) {
            // for (auto&& af : attrs_node->ListFieldInfo()) {
            //     std::cout << PrettyPrint(af->name) << std::endl;
            // }
            if (op_node->name == "Comm") {
                auto comm_node = call->attrs.as<CommAttrs>();
                // std::cout << comm_node->comm_target << " " << comm_node->sub_type << " " << comm_node->comm_tag << " " << comm_node->comm_size_in_KB << std::endl;
                op_decl_stream << comm_node->comm_target << ", " << comm_node->sub_type << ", " << comm_node->comm_tag << ", " << comm_node->comm_size_in_KB;
            }
            else if (op_node->name == "nn.conv2d") {
                auto conv_node = call->attrs.as<Conv2DAttrs>();
                for (auto&& ks : conv_node->kernel_size) {
                    op_decl_stream << PrettyPrint(ks) << ", ";
                }
                for (auto&& pd : conv_node->padding) {
                    op_decl_stream << PrettyPrint(pd) << ", ";
                }
                for (auto&& st : conv_node->strides) {
                    op_decl_stream << PrettyPrint(st) << ", ";
                }
                for (auto&& dl : conv_node->dilation) {
                    op_decl_stream << PrettyPrint(dl) << ", ";
                }
                op_decl_stream << conv_node->groups;
            }
        }
        op_decl_stream << ");\n";
        for (size_t i = 0; i < call->args.size(); i++) {
            VisitExpr(call->args[i]);
        }
    }
    // std::string JIT() { return "JIT"; }
private:
    std::string ext_func_id_ = "";
    int op_idx = 0;
    int buf_idx_ = 0;
    int tensor_idx = 0;
    std::vector<std::string> ext_func_args_;
    std::vector<std::string> ext_func_body;
    std::vector<std::string> func_decl_;
    std::vector<std::string> buf_decl_;
    std::map<size_t, int> tensor_idx_map;
    Array<String> const_vars_;
    std::vector<std::pair<std::string, int>> out_;
    friend class TianMuSourceCodegen;
};

class TianMuSourceCodegen : public TianMuSourceModuleCodegenBase {
public:
    std::pair<std::string, Array<String>> GenTianMuFunc(const Function& func) { 
        auto sid = GetExtSymbol(func);
        CodegenTianMu builder(sid);
        builder.VisitExpr(func->body);
        std::cout << builder.getTensorIdx() << std::endl;
        std::cout << builder.op_decl_stream.str() << std::endl;
        return {sid, builder.const_vars_};
    }
    runtime::Module CreateTianMuSourceModule(const ObjectRef& ref) override { 
        std::cout << "CodeGen TianMu CreateTianMuSourceModule" << std::endl;
        ICHECK(ref->IsInstance<FunctionNode>());
        auto res = GenTianMuFunc(Downcast<Function>(ref));
        std::string code = "Code Generating ...";
        code_stream_ << "#include <iostream>\n";
        code_stream_ << "#include \"tianmu.h\"\n";
        String sym = std::get<0>(res);
        Array<String> variables = std::get<1>(res);
        const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
        ICHECK(pf != nullptr) << "pf NULL" << std::endl;
        return (*pf)(code.c_str(), "c", sym, variables);
    }
private:
  std::ostringstream code_stream_;
};

runtime::Module TianMuCompiler(const ObjectRef& ref) {
    TianMuSourceCodegen csrc;
    return csrc.CreateTianMuSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.tianmu_compiler").set_body_typed(TianMuCompiler);

}
}
}