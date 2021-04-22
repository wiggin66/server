// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include "src/clients/c++/perf_analyzer/c_api_helpers/triton_loader.h"
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <fstream>
#include <string>
#include "src/clients/c++/perf_analyzer/c_api_helpers/common.h"

#include <thread>

namespace cb = perfanalyzer::clientbackend;
namespace perfanalyzer { namespace clientbackend {
namespace {
bool enforce_memory_type = false;
TRITONSERVER_MemoryType requested_memory_type;
/// Helper function for allocating memory
TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  // Initially attempt to make the actual memory type and id that we
  // allocate be the same as preferred memory type
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    std::cout << "allocated " << byte_size << " bytes for result tensor "
              << tensor_name << std::endl;
  } else {
    void* allocated_ptr = nullptr;
    if (enforce_memory_type) {
      *actual_memory_type = requested_memory_type;
    }

    switch (*actual_memory_type) {
      // Use CPU memory if the requested memory type is unknown
      // (default case).
      case TRITONSERVER_MEMORY_CPU:
      default: {
        *actual_memory_type = TRITONSERVER_MEMORY_CPU;
        allocated_ptr = malloc(byte_size);
        break;
      }
    }

    // Pass the tensor name with buffer_userp so we can show it when
    // releasing the buffer.
    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = new std::string(tensor_name);
      std::cout << "allocated " << byte_size << " bytes in "
                << size_t(*actual_memory_type) << " for result tensor "
                << tensor_name << std::endl;
    }
  }

  return nullptr;  // Success
}

/// Helper function for allocating memory
TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  std::string* name = nullptr;
  if (buffer_userp != nullptr) {
    name = reinterpret_cast<std::string*>(buffer_userp);
  } else {
    name = new std::string("<unknown>");
  }

  std::cout << "Releasing buffer " << buffer << " of size " << byte_size
            << " in " << size_t(memory_type) << " for result '" << *name << "'"
            << std::endl;
  switch (memory_type) {
    case TRITONSERVER_MEMORY_CPU:
      free(buffer);
      break;
    default:
      std::cerr << "error: unexpected buffer allocated in CUDA managed memory"
                << std::endl;
      break;
  }

  delete name;

  return nullptr;  // Success
}

}  // namespace
Error
TritonLoader::Create(
    const std::string& library_directory, const std::string& model_repository,
    const std::string& memory_type, std::shared_ptr<TritonLoader>* loader)
{
  if (library_directory.empty() || model_repository.empty()) {
    return Error("cannot load server, paths are empty");
  }
  std::shared_ptr<TritonLoader> triton_loader = std::make_shared<TritonLoader>(
      library_directory, model_repository, memory_type);

  Error status = triton_loader->LoadServerLibrary();
  assert(status.IsOk());
  status = triton_loader->StartTriton(memory_type, false);
  assert(status.IsOk());

  *loader = std::move(triton_loader);
  return Error::Success;
}

TritonLoader::TritonLoader(
    const std::string& library_directory, const std::string& model_repository,
    const std::string& memory_type)
{
  library_directory_ = library_directory;
  model_repository_path_ = model_repository;
}

Error
TritonLoader::StartTriton(const std::string& memory_type, bool isVerbose)
{
  if (!memory_type.empty()) {
    enforce_memory_type_ = true;
    if (memory_type.compare("system")) {
      requested_memory_type_ = TRITONSERVER_MEMORY_CPU;
    } else if (memory_type.compare("pinned")) {
      requested_memory_type_ = TRITONSERVER_MEMORY_CPU_PINNED;
    } else if (memory_type.compare("gpu")) {
      requested_memory_type_ = TRITONSERVER_MEMORY_GPU;
    } else {
      return Error("Specify one of the following types: system, pinned or gpu");
    }
    requested_memory_type = requested_memory_type_;
    enforce_memory_type = enforce_memory_type_;
  }

  if (isVerbose) {
    verbose_level_ = 1;
  }

  // Check API version.
  uint32_t api_version_major, api_version_minor;
  REPORT_TRITONSERVER_ERROR(
      api_version_fn_(&api_version_major, &api_version_minor));
  std::cout << "api version major: " << api_version_major
            << ", minor: " << api_version_minor << std::endl;
  if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
      (TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
    return Error("triton server API version mismatch");
  }

  // Create the server...
  TRITONSERVER_ServerOptions* server_options = nullptr;
  RETURN_IF_TRITONSERVER_ERROR(
      options_new_fn_(&server_options), "creating server options");
  RETURN_IF_TRITONSERVER_ERROR(
      options_set_model_repo_path_fn_(
          server_options, model_repository_path_.c_str()),
      "setting model repository path");
  RETURN_IF_TRITONSERVER_ERROR(
      set_log_verbose_fn_(server_options, verbose_level_),
      "setting verbose logging level");
  RETURN_IF_TRITONSERVER_ERROR(
      set_backend_directory_fn_(
          server_options, (library_directory_ + "/backends").c_str()),
      "setting backend directory");
  RETURN_IF_TRITONSERVER_ERROR(
      set_repo_agent_directory_fn_(
          server_options, (library_directory_ + "/repoagents").c_str()),
      "setting repository agent directory");
  RETURN_IF_TRITONSERVER_ERROR(
      set_strict_model_config_fn_(server_options, true),
      "setting strict model configuration");
  double min_compute_capability = 0;
  // FIXME: Do not have GPU support right now
  RETURN_IF_TRITONSERVER_ERROR(
      set_min_supported_compute_capability_fn_(
          server_options, min_compute_capability),
      "setting minimum supported CUDA compute capability");
  RETURN_IF_TRITONSERVER_ERROR(
      server_new_fn_(&server_ptr_, server_options), "creating server");
  RETURN_IF_TRITONSERVER_ERROR(
      server_options_delete_fn_(server_options), "deleting server options");
  std::shared_ptr<TRITONSERVER_Server> shared_server(
      server_ptr_, server_delete_fn_);
  server_ = shared_server;

  // Wait until the server is both live and ready.
  size_t health_iters = 0;
  while (true) {
    bool live, ready;
    RETURN_IF_TRITONSERVER_ERROR(
        server_is_live_fn_(server_.get(), &live),
        "unable to get server liveness");
    RETURN_IF_TRITONSERVER_ERROR(
        server_is_ready_fn_(server_.get(), &ready),
        "unable to get server readiness");
    std::cout << "Server Health: live " << live << ", ready " << ready
              << std::endl;
    if (live && ready) {
      std::cout << "server is alive!" << std::endl;
      break;
    }

    if (++health_iters >= 10) {
      return Error("failed to find healthy inference server");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  // Print status of the server.
  {
    TRITONSERVER_Message* server_metadata_message;
    RETURN_IF_TRITONSERVER_ERROR(
        server_metadata_fn_(server_.get(), &server_metadata_message),
        "unable to get server metadata message");
    const char* buffer;
    size_t byte_size;
    RETURN_IF_TRITONSERVER_ERROR(
        message_serialize_to_json_fn_(
            server_metadata_message, &buffer, &byte_size),
        "unable to serialize server metadata message");

    std::cout << "Server Status:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;

    RETURN_IF_TRITONSERVER_ERROR(
        message_delete_fn_(server_metadata_message),
        "deleting status metadata");
  }
  server_is_ready_ = true;

  return Error::Success;
}

Error
TritonLoader::ServerMetaData(rapidjson::Document* server_metadata) const
{
  if (!ServerIsReady()) {
    return Error("Model is not loaded and/or server is not ready");
  }
  std::cout << "ServerMetaData..." << std::endl;

  TRITONSERVER_Message* server_metadata_message;
  RETURN_IF_TRITONSERVER_ERROR(
      server_metadata_fn_(server_.get(), &server_metadata_message),
      "unable to get server metadata message");
  const char* buffer;
  size_t byte_size;
  RETURN_IF_TRITONSERVER_ERROR(
      message_serialize_to_json_fn_(
          server_metadata_message, &buffer, &byte_size),
      "unable to serialize server metadata message");
  server_metadata->Parse(buffer, byte_size);
  if (server_metadata->HasParseError()) {
    return Error(
        "error: failed to parse server metadata from JSON: " +
        std::string(GetParseError_En(server_metadata->GetParseError())) +
        " at " + std::to_string(server_metadata->GetErrorOffset()));
  }
  RETURN_IF_TRITONSERVER_ERROR(
      message_delete_fn_(server_metadata_message), "deleting status metadata");
  return Error::Success;
}

Error
TritonLoader::LoadModel(
    const std::string& model_name, const std::string& model_version)
{
  std::cout << "loading model..." << std::endl;
  if (!ServerIsReady()) {
    return Error("server is not ready, abort!");
  }
  model_name_ = model_name;

  RETURN_IF_ERROR(GetModelVersionFromString(model_version, &model_version_));
  // Wait for the model to become available.
  bool is_ready = false;
  size_t health_iters = 0;

  // some error handling
  if (model_repository_path_.empty()) {
    return Error("Need to specify model repository");
  }
  while (!is_ready) {
    RETURN_IF_TRITONSERVER_ERROR(
        model_is_ready_fn_(
            server_.get(), model_name_.c_str(), model_version_, &is_ready),
        "unable to get model readiness");
    if (!is_ready) {
      if (++health_iters >= 10) {
        return Error("model failed to be ready in 10 iterations");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }
  }

  // Create the allocator that will be used to allocate buffers for
  // the result tensors.
  RETURN_IF_TRITONSERVER_ERROR(
      response_allocator_new_fn_(
          &allocator_,
          reinterpret_cast<
              TRITONSERVER_Error* (*)(TRITONSERVER_ResponseAllocator * allocator, const char* tensor_name, size_t byte_size, TRITONSERVER_MemoryType memory_type, int64_t memory_type_id, void* userp, void** buffer, void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type, int64_t* actual_memory_type_id)>(
              ResponseAlloc),
          reinterpret_cast<
              TRITONSERVER_Error* (*)(TRITONSERVER_ResponseAllocator * allocator, void* buffer, void* buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)>(
              ResponseRelease),
          nullptr /* start_fn */),
      "creating response allocator");

  std::cout << "loaded model " << model_name_ << std::endl;
  model_is_loaded_ = true;  // flag to confirm model is correct and loaded
  return Error::Success;
}

Error
TritonLoader::ModelMetadata(rapidjson::Document* model_metadata) const
{
  if (!ModelIsLoaded() || !ServerIsReady()) {
    return Error("Model is not loaded and/or server is not ready");
  }
  std::cout << "ModelMetadata..." << std::endl;
  TRITONSERVER_Message* model_metadata_message;

  // get model metadata
  RETURN_IF_TRITONSERVER_ERROR(
      model_metadata_fn_(
          server_.get(), model_name_.c_str(), model_version_,
          &model_metadata_message),
      "unable to get model metadata message");
  const char* buffer;
  size_t byte_size;
  RETURN_IF_TRITONSERVER_ERROR(
      message_serialize_to_json_fn_(
          model_metadata_message, &buffer, &byte_size),
      "unable to serialize model status protobuf");

  model_metadata->Parse(buffer, byte_size);
  if (model_metadata->HasParseError()) {
    return Error(
        "error: failed to parse model metadata from JSON: " +
        std::string(GetParseError_En(model_metadata->GetParseError())) +
        " at " + std::to_string(model_metadata->GetErrorOffset()));
  }

  RETURN_IF_TRITONSERVER_ERROR(
      message_delete_fn_(model_metadata_message), "deleting status protobuf");

  if (strcmp((*model_metadata)["name"].GetString(), model_name_.c_str())) {
    return Error("unable to find metadata for model");
  }

  bool found_version = false;
  if (model_metadata->HasMember("versions")) {
    for (const auto& version : (*model_metadata)["versions"].GetArray()) {
      if (strcmp(version.GetString(), std::to_string(model_version_).c_str()) ==
          0) {
        found_version = true;
        break;
      }
    }
  }
  if (!found_version) {
    std::string msg = "unable to find version " +
                      std::to_string(model_version_) + " status for model";
    return Error(msg);
  }
  return Error::Success;
}

Error
TritonLoader::ModelConfig(rapidjson::Document* model_config) const
{
  if (!ModelIsLoaded() || !ServerIsReady()) {
    return Error("Model is not loaded and/or server is not ready");
  }
  std::cout << "ModelConfig..." << std::endl;

  TRITONSERVER_Message* model_config_message;
  uint32_t config_version = 1;
  RETURN_IF_TRITONSERVER_ERROR(
      model_config_fn_(
          server_.get(), model_name_.c_str(), model_version_, config_version,
          &model_config_message),
      "unable to get model config message");
  const char* buffer;
  size_t byte_size;
  RETURN_IF_TRITONSERVER_ERROR(
      message_serialize_to_json_fn_(model_config_message, &buffer, &byte_size),
      "unable to serialize model config status protobuf");

  model_config->Parse(buffer, byte_size);
  if (model_config->HasParseError()) {
    return Error(
        "error: failed to parse model config from JSON: " +
        std::string(GetParseError_En(model_config->GetParseError())) + " at " +
        std::to_string(model_config->GetErrorOffset()));
  }

  RETURN_IF_TRITONSERVER_ERROR(
      message_delete_fn_(model_config_message),
      "deleting server config status protobuf");

  return Error::Success;
}

Error
TritonLoader::LoadServerLibrary()
{
  std::string full_path = library_directory_ + SERVER_LIBRARY_PATH;
  RETURN_IF_ERROR(FileExists(full_path));
  FAIL_IF_ERR(
      OpenLibraryHandle(full_path, &dlhandle_),
      "shared library loading library:" + full_path);

  TritonServerApiVersionFn_t apifn;
  TritonServerOptionsNewFn_t onfn;
  TritonServerOptionSetModelRepoPathFn_t rpfn;
  TritonServerSetLogVerboseFn_t slvfn;

  TritonServerSetBackendDirFn_t sbdfn;
  TritonServerSetRepoAgentDirFn_t srdfn;
  TritonServerSetStrictModelConfigFn_t ssmcfn;
  TritonServerSetMinSupportedComputeCapabilityFn_t smsccfn;

  TritonServerNewFn_t snfn;
  TritonServerOptionsDeleteFn_t odfn;
  TritonServerDeleteFn_t sdfn;
  TritonServerIsLiveFn_t ilfn;

  TritonServerIsReadyFn_t irfn;
  TritonServerMetadataFn_t smfn;
  TritonServerMessageSerializeToJsonFn_t stjfn;
  TritonServerMessageDeleteFn_t mdfn;

  TritonServerModelIsReadyFn_t mirfn;
  TritonServerModelMetadataFn_t mmfn;
  TritonServerResponseAllocatorNewFn_t ranfn;
  TritonServerInferenceRequestNewFn_t irnfn;

  TritonServerInferenceRequestSetIdFn_t irsifn;
  TritonServerInferenceRequestSetReleaseCallbackFn_t irsrcfn;
  TritonServerInferenceRequestAddInputFn_t iraifn;
  TritonServerInferenceRequestAddRequestedOutputFn_t irarofn;

  TritonServerInferenceRequestAppendInputDataFn_t iraidfn;
  TritonServerInferenceRequestSetResponseCallbackFn_t irsrescfn;
  TritonServerInferAsyncFn_t iafn;
  TritonServerInferenceResponseErrorFn_t irefn;

  TritonServerInferenceResponseDeleteFn_t irdfn;
  TritonServerInferenceRequestRemoveAllInputDataFn_t irraidfn;
  TritonServerResponseAllocatorDeleteFn_t iradfn;
  TritonServerErrorNewFn_t enfn;

  TritonServerMemoryTypeStringFn_t mtsfn;
  TritonServerInferenceResponseOutputCountFn_t irocfn;
  TritonServerDataTypeStringFn_t dtsfn;
  TritonServerErrorMessageFn_t emfn;
  TritonServerErrorDeleteFn_t edfn;
  TritonServerErrorCodeToStringFn_t ectsfn;
  TritonServerModelConfigFn_t mcfn;

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ApiVersion", true /* optional */,
      reinterpret_cast<void**>(&apifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsNew", true /* optional */,
      reinterpret_cast<void**>(&onfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetModelRepositoryPath",
      true /* optional */, reinterpret_cast<void**>(&rpfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetLogVerbose", true /* optional */,
      reinterpret_cast<void**>(&slvfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetBackendDirectory",
      true /* optional */, reinterpret_cast<void**>(&sbdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetRepoAgentDirectory",
      true /* optional */, reinterpret_cast<void**>(&srdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetStrictModelConfig",
      true /* optional */, reinterpret_cast<void**>(&ssmcfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability",
      true /* optional */, reinterpret_cast<void**>(&smsccfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerNew", true /* optional */,
      reinterpret_cast<void**>(&snfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsDelete", true /* optional */,
      reinterpret_cast<void**>(&odfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerDelete", true /* optional */,
      reinterpret_cast<void**>(&sdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerIsLive", true /* optional */,
      reinterpret_cast<void**>(&ilfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerIsReady", true /* optional */,
      reinterpret_cast<void**>(&irfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerMetadata", true /* optional */,
      reinterpret_cast<void**>(&smfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MessageSerializeToJson", true /* optional */,
      reinterpret_cast<void**>(&stjfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MessageDelete", true /* optional */,
      reinterpret_cast<void**>(&mdfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelIsReady", true /* optional */,
      reinterpret_cast<void**>(&mirfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelMetadata", true /* optional */,
      reinterpret_cast<void**>(&mmfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ResponseAllocatorNew", true /* optional */,
      reinterpret_cast<void**>(&ranfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestNew", true /* optional */,
      reinterpret_cast<void**>(&irnfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetId", true /* optional */,
      reinterpret_cast<void**>(&irsifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetReleaseCallback",
      true /* optional */, reinterpret_cast<void**>(&irsrcfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAddInput", true /* optional */,
      reinterpret_cast<void**>(&iraifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAddRequestedOutput",
      true /* optional */, reinterpret_cast<void**>(&irarofn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAppendInputData",
      true /* optional */, reinterpret_cast<void**>(&iraidfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetResponseCallback",
      true /* optional */, reinterpret_cast<void**>(&irsrescfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerInferAsync", true /* optional */,
      reinterpret_cast<void**>(&iafn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseError", true /* optional */,
      reinterpret_cast<void**>(&irefn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseDelete", true /* optional */,
      reinterpret_cast<void**>(&irdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestRemoveAllInputData",
      true /* optional */, reinterpret_cast<void**>(&irraidfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ResponseAllocatorDelete", true /* optional */,
      reinterpret_cast<void**>(&iradfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorNew", true /* optional */,
      reinterpret_cast<void**>(&enfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MemoryTypeString", true /* optional */,
      reinterpret_cast<void**>(&mtsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseOutputCount",
      true /* optional */, reinterpret_cast<void**>(&irocfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_DataTypeString", true /* optional */,
      reinterpret_cast<void**>(&dtsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorMessage", true /* optional */,
      reinterpret_cast<void**>(&emfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorDelete", true /* optional */,
      reinterpret_cast<void**>(&edfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorCodeString", true /* optional */,
      reinterpret_cast<void**>(&ectsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelConfig", true /* optional */,
      reinterpret_cast<void**>(&mcfn)));

  api_version_fn_ = apifn;
  options_new_fn_ = onfn;
  options_set_model_repo_path_fn_ = rpfn;
  set_log_verbose_fn_ = slvfn;

  set_backend_directory_fn_ = sbdfn;
  set_repo_agent_directory_fn_ = srdfn;
  set_strict_model_config_fn_ = ssmcfn;
  set_min_supported_compute_capability_fn_ = smsccfn;

  server_new_fn_ = snfn;
  server_options_delete_fn_ = odfn;
  server_delete_fn_ = sdfn;
  server_is_live_fn_ = ilfn;

  server_is_ready_fn_ = irfn;
  server_metadata_fn_ = smfn;
  message_serialize_to_json_fn_ = stjfn;
  message_delete_fn_ = mdfn;

  model_is_ready_fn_ = mirfn;
  model_metadata_fn_ = mmfn;
  response_allocator_new_fn_ = ranfn;
  inference_request_new_fn_ = irnfn;

  inference_request_set_id_fn_ = irsifn;
  inference_request_set_release_callback_fn_ = irsrcfn;
  inference_request_add_input_fn_ = iraifn;
  inference_request_add_requested_output_fn_ = irarofn;

  inference_request_append_input_data_fn_ = iraidfn;
  inference_request_set_response_callback_fn_ = irsrescfn;
  infer_async_fn_ = iafn;
  inference_response_error_fn_ = irefn;

  inference_response_delete_fn_ = irdfn;
  inference_request_remove_all_input_data_fn_ = irraidfn;
  response_allocator_delete_fn_ = iradfn;
  error_new_fn_ = enfn;

  memory_type_string_fn_ = mtsfn;
  inference_response_output_count_fn_ = irocfn;
  data_type_string_fn_ = dtsfn;
  error_message_fn_ = emfn;
  error_delete_fn_ = edfn;
  error_code_to_string_fn_ = ectsfn;
  model_config_fn_ = mcfn;

  return Error::Success;
}

void
TritonLoader::ClearHandles()
{
  dlhandle_ = nullptr;

  api_version_fn_ = nullptr;
  options_new_fn_ = nullptr;
  options_set_model_repo_path_fn_ = nullptr;
  set_log_verbose_fn_ = nullptr;

  set_backend_directory_fn_ = nullptr;
  set_repo_agent_directory_fn_ = nullptr;
  set_strict_model_config_fn_ = nullptr;
  set_min_supported_compute_capability_fn_ = nullptr;

  server_new_fn_ = nullptr;
  server_options_delete_fn_ = nullptr;
  server_delete_fn_ = nullptr;
  server_is_live_fn_ = nullptr;

  server_is_ready_fn_ = nullptr;
  server_metadata_fn_ = nullptr;
  message_serialize_to_json_fn_ = nullptr;
  message_delete_fn_ = nullptr;

  model_is_ready_fn_ = nullptr;
  model_metadata_fn_ = nullptr;
  response_allocator_new_fn_ = nullptr;
  inference_request_new_fn_ = nullptr;

  inference_request_set_id_fn_ = nullptr;
  inference_request_set_release_callback_fn_ = nullptr;
  inference_request_add_input_fn_ = nullptr;
  inference_request_add_requested_output_fn_ = nullptr;

  inference_request_append_input_data_fn_ = nullptr;
  inference_request_set_response_callback_fn_ = nullptr;
  infer_async_fn_ = nullptr;
  inference_response_error_fn_ = nullptr;

  inference_response_delete_fn_ = nullptr;
  inference_request_remove_all_input_data_fn_ = nullptr;
  response_allocator_delete_fn_ = nullptr;
  error_new_fn_ = nullptr;

  memory_type_string_fn_ = nullptr;
  inference_response_output_count_fn_ = nullptr;
  data_type_string_fn_ = nullptr;
  error_message_fn_ = nullptr;
  error_delete_fn_ = nullptr;
  error_code_to_string_fn_ = nullptr;

  options_ = nullptr;
  server_ptr_ = nullptr;
  allocator_ = nullptr;
}

Error
TritonLoader::FileExists(std::string& filepath)
{
  std::ifstream ifile;
  ifile.open(filepath);
  if (!ifile) {
    return Error("unable to find local Triton library: " + filepath);
  } else {
    return Error::Success;
  }
}


/// delete
TRITONSERVER_Error*
TritonLoader::ParseModelMetadata(
    const rapidjson::Document& model_metadata, bool* is_int,
    bool* is_torch_model)
{
  std::string seen_data_type;
  for (const auto& input : model_metadata["inputs"].GetArray()) {
    if (strcmp(input["datatype"].GetString(), "INT32") &&
        strcmp(input["datatype"].GetString(), "FP32")) {
      return error_new_fn_(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "simple lib example only supports model with data type INT32 or "
          "FP32");
    }
    if (seen_data_type.empty()) {
      seen_data_type = input["datatype"].GetString();
    } else if (strcmp(seen_data_type.c_str(), input["datatype"].GetString())) {
      return error_new_fn_(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of 'simple' model must have the data type");
    }
  }
  for (const auto& output : model_metadata["outputs"].GetArray()) {
    if (strcmp(output["datatype"].GetString(), "INT32") &&
        strcmp(output["datatype"].GetString(), "FP32")) {
      return error_new_fn_(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "simple lib example only supports model with data type INT32 or "
          "FP32");
    } else if (strcmp(seen_data_type.c_str(), output["datatype"].GetString())) {
      return error_new_fn_(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of 'simple' model must have the data type");
    }
  }

  *is_int = (strcmp(seen_data_type.c_str(), "INT32") == 0);
  *is_torch_model =
      (strcmp(model_metadata["platform"].GetString(), "pytorch_libtorch") == 0);
  return nullptr;
}


}}  // namespace perfanalyzer::clientbackend