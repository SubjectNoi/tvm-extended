/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file registry.cc
 * \brief The global registry of packed function.
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/logging.h>

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "runtime_base.h"

namespace tvm {
namespace runtime {

struct Registry::Manager {
  // map storing the functions.
  // We delibrately used raw pointer
  // This is because PackedFunc can contain callbacks into the host languge(python)
  // and the resource can become invalid because of indeterminstic order of destruction and forking.
  // The resources will only be recycled during program exit.
  std::unordered_map<std::string, Registry*> fmap;
  // mutex
  std::mutex mutex;

  Manager() {}

  static Manager* Global() {
    // We deliberately leak the Manager instance, to avoid leak sanitizers
    // complaining about the entries in Manager::fmap being leaked at program
    // exit.
    static Manager* inst = new Manager();
    return inst;
  }
};

Registry& Registry::set_body(PackedFunc f) {  // NOLINT(*)
  func_ = f;
  return *this;
}

Registry& Registry::Register(const std::string& name, bool can_override) {  // NOLINT(*)
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  if (m->fmap.count(name)) {
    // std::cout << name << " " << can_override << std::endl;
    if (name == "relay.op._make.Comm") can_override = true;
    ICHECK(can_override) << "Global PackedFunc " << name << " is already registered";
  }

  Registry* r = new Registry();
  r->name_ = name;
  m->fmap[name] = r;
  return *r;
}

bool Registry::Remove(const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) return false;
  m->fmap.erase(it);
  return true;
}

const PackedFunc* Registry::Get(const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) return nullptr;
  return &(it->second->func_);
}

std::vector<std::string> Registry::ListNames() {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  std::vector<std::string> keys;
  keys.reserve(m->fmap.size());
  for (const auto& kv : m->fmap) {
    keys.push_back(kv.first);
  }
  return keys;
}

}  // namespace runtime
}  // namespace tvm

/*! \brief entry to to easily hold returning information */
struct TVMFuncThreadLocalEntry {
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char*> ret_vec_charp;
};

/*! \brief Thread local store that can be used to hold return values. */
typedef dmlc::ThreadLocalStore<TVMFuncThreadLocalEntry> TVMFuncThreadLocalStore;

int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) {
  API_BEGIN();
  tvm::runtime::Registry::Register(name, override != 0)
      .set_body(*static_cast<tvm::runtime::PackedFunc*>(f));
  API_END();
}

int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out) {
  API_BEGIN();
  const tvm::runtime::PackedFunc* fp = tvm::runtime::Registry::Get(name);
  if (fp != nullptr) {
    *out = new tvm::runtime::PackedFunc(*fp);  // NOLINT(*)
  } else {
    *out = nullptr;
  }
  API_END();
}

int TVMFuncListGlobalNames(int* out_size, const char*** out_array) {
  API_BEGIN();
  TVMFuncThreadLocalEntry* ret = TVMFuncThreadLocalStore::Get();
  ret->ret_vec_str = tvm::runtime::Registry::ListNames();
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_array = dmlc::BeginPtr(ret->ret_vec_charp);
  *out_size = static_cast<int>(ret->ret_vec_str.size());
  API_END();
}

int TVMFuncRemoveGlobal(const char* name) {
  API_BEGIN();
  tvm::runtime::Registry::Remove(name);
  API_END();
}
