//! Mirrors Python's CallTypes and PassthroughCallTypes (litellm/types/utils.py), generated from their variant lists.

// variant names mirror Python's enum members verbatim
#![allow(non_camel_case_types)]

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum CallTypes {
    #[strum(serialize = "embedding")]
    embedding,
    #[strum(serialize = "aembedding")]
    aembedding,
    #[strum(serialize = "completion")]
    completion,
    #[strum(serialize = "acompletion")]
    acompletion,
    #[strum(serialize = "atext_completion")]
    atext_completion,
    #[strum(serialize = "text_completion")]
    text_completion,
    #[strum(serialize = "image_generation")]
    image_generation,
    #[strum(serialize = "aimage_generation")]
    aimage_generation,
    #[strum(serialize = "image_edit")]
    image_edit,
    #[strum(serialize = "aimage_edit")]
    aimage_edit,
    #[strum(serialize = "moderation")]
    moderation,
    #[strum(serialize = "amoderation")]
    amoderation,
    #[strum(serialize = "atranscription")]
    atranscription,
    #[strum(serialize = "transcription")]
    transcription,
    #[strum(serialize = "aspeech")]
    aspeech,
    #[strum(serialize = "speech")]
    speech,
    #[strum(serialize = "rerank")]
    rerank,
    #[strum(serialize = "arerank")]
    arerank,
    #[strum(serialize = "search")]
    search,
    #[strum(serialize = "asearch")]
    asearch,
    #[strum(serialize = "_arealtime")]
    arealtime,
    #[strum(serialize = "_aresponses_websocket")]
    aresponses_websocket,
    #[strum(serialize = "create_batch")]
    create_batch,
    #[strum(serialize = "acreate_batch")]
    acreate_batch,
    #[strum(serialize = "aretrieve_batch")]
    aretrieve_batch,
    #[strum(serialize = "retrieve_batch")]
    retrieve_batch,
    #[strum(serialize = "acancel_batch")]
    acancel_batch,
    #[strum(serialize = "cancel_batch")]
    cancel_batch,
    #[strum(serialize = "pass_through_endpoint")]
    pass_through,
    #[strum(serialize = "anthropic_messages")]
    anthropic_messages,
    #[strum(serialize = "aanthropic_messages")]
    aanthropic_messages,
    #[strum(serialize = "get_assistants")]
    get_assistants,
    #[strum(serialize = "aget_assistants")]
    aget_assistants,
    #[strum(serialize = "create_assistants")]
    create_assistants,
    #[strum(serialize = "acreate_assistants")]
    acreate_assistants,
    #[strum(serialize = "delete_assistant")]
    delete_assistant,
    #[strum(serialize = "adelete_assistant")]
    adelete_assistant,
    #[strum(serialize = "acreate_thread")]
    acreate_thread,
    #[strum(serialize = "create_thread")]
    create_thread,
    #[strum(serialize = "aget_thread")]
    aget_thread,
    #[strum(serialize = "get_thread")]
    get_thread,
    #[strum(serialize = "a_add_message")]
    a_add_message,
    #[strum(serialize = "add_message")]
    add_message,
    #[strum(serialize = "aget_messages")]
    aget_messages,
    #[strum(serialize = "get_messages")]
    get_messages,
    #[strum(serialize = "arun_thread")]
    arun_thread,
    #[strum(serialize = "run_thread")]
    run_thread,
    #[strum(serialize = "arun_thread_stream")]
    arun_thread_stream,
    #[strum(serialize = "run_thread_stream")]
    run_thread_stream,
    #[strum(serialize = "afile_retrieve")]
    afile_retrieve,
    #[strum(serialize = "file_retrieve")]
    file_retrieve,
    #[strum(serialize = "afile_delete")]
    afile_delete,
    #[strum(serialize = "file_delete")]
    file_delete,
    #[strum(serialize = "afile_list")]
    afile_list,
    #[strum(serialize = "file_list")]
    file_list,
    #[strum(serialize = "acreate_file")]
    acreate_file,
    #[strum(serialize = "create_file")]
    create_file,
    #[strum(serialize = "afile_content")]
    afile_content,
    #[strum(serialize = "file_content")]
    file_content,
    #[strum(serialize = "create_fine_tuning_job")]
    create_fine_tuning_job,
    #[strum(serialize = "acreate_fine_tuning_job")]
    acreate_fine_tuning_job,
    #[strum(serialize = "create_video")]
    create_video,
    #[strum(serialize = "acreate_video")]
    acreate_video,
    #[strum(serialize = "video_generation")]
    video_generation,
    #[strum(serialize = "avideo_generation")]
    avideo_generation,
    #[strum(serialize = "avideo_retrieve")]
    avideo_retrieve,
    #[strum(serialize = "video_retrieve")]
    video_retrieve,
    #[strum(serialize = "avideo_content")]
    avideo_content,
    #[strum(serialize = "video_content")]
    video_content,
    #[strum(serialize = "video_remix")]
    video_remix,
    #[strum(serialize = "avideo_remix")]
    avideo_remix,
    #[strum(serialize = "video_list")]
    video_list,
    #[strum(serialize = "avideo_list")]
    avideo_list,
    #[strum(serialize = "video_retrieve_job")]
    video_retrieve_job,
    #[strum(serialize = "avideo_retrieve_job")]
    avideo_retrieve_job,
    #[strum(serialize = "video_delete")]
    video_delete,
    #[strum(serialize = "avideo_delete")]
    avideo_delete,
    #[strum(serialize = "video_create_character")]
    video_create_character,
    #[strum(serialize = "avideo_create_character")]
    avideo_create_character,
    #[strum(serialize = "video_get_character")]
    video_get_character,
    #[strum(serialize = "avideo_get_character")]
    avideo_get_character,
    #[strum(serialize = "video_edit")]
    video_edit,
    #[strum(serialize = "avideo_edit")]
    avideo_edit,
    #[strum(serialize = "video_extension")]
    video_extension,
    #[strum(serialize = "avideo_extension")]
    avideo_extension,
    #[strum(serialize = "vector_store_file_create")]
    vector_store_file_create,
    #[strum(serialize = "avector_store_file_create")]
    avector_store_file_create,
    #[strum(serialize = "vector_store_file_list")]
    vector_store_file_list,
    #[strum(serialize = "avector_store_file_list")]
    avector_store_file_list,
    #[strum(serialize = "vector_store_file_retrieve")]
    vector_store_file_retrieve,
    #[strum(serialize = "avector_store_file_retrieve")]
    avector_store_file_retrieve,
    #[strum(serialize = "vector_store_file_content")]
    vector_store_file_content,
    #[strum(serialize = "avector_store_file_content")]
    avector_store_file_content,
    #[strum(serialize = "vector_store_file_update")]
    vector_store_file_update,
    #[strum(serialize = "avector_store_file_update")]
    avector_store_file_update,
    #[strum(serialize = "vector_store_file_delete")]
    vector_store_file_delete,
    #[strum(serialize = "avector_store_file_delete")]
    avector_store_file_delete,
    #[strum(serialize = "vector_store_create")]
    vector_store_create,
    #[strum(serialize = "avector_store_create")]
    avector_store_create,
    #[strum(serialize = "vector_store_search")]
    vector_store_search,
    #[strum(serialize = "avector_store_search")]
    avector_store_search,
    #[strum(serialize = "ingest")]
    ingest,
    #[strum(serialize = "aingest")]
    aingest,
    #[strum(serialize = "query")]
    query,
    #[strum(serialize = "aquery")]
    aquery,
    #[strum(serialize = "create_interaction")]
    create_interaction,
    #[strum(serialize = "acreate_interaction")]
    acreate_interaction,
    #[strum(serialize = "create_container")]
    create_container,
    #[strum(serialize = "acreate_container")]
    acreate_container,
    #[strum(serialize = "list_containers")]
    list_containers,
    #[strum(serialize = "alist_containers")]
    alist_containers,
    #[strum(serialize = "retrieve_container")]
    retrieve_container,
    #[strum(serialize = "aretrieve_container")]
    aretrieve_container,
    #[strum(serialize = "delete_container")]
    delete_container,
    #[strum(serialize = "adelete_container")]
    adelete_container,
    #[strum(serialize = "list_container_files")]
    list_container_files,
    #[strum(serialize = "alist_container_files")]
    alist_container_files,
    #[strum(serialize = "upload_container_file")]
    upload_container_file,
    #[strum(serialize = "aupload_container_file")]
    aupload_container_file,
    #[strum(serialize = "create_sandbox")]
    create_sandbox,
    #[strum(serialize = "acreate_sandbox")]
    acreate_sandbox,
    #[strum(serialize = "delete_sandbox")]
    delete_sandbox,
    #[strum(serialize = "adelete_sandbox")]
    adelete_sandbox,
    #[strum(serialize = "run_code")]
    run_code,
    #[strum(serialize = "arun_code")]
    arun_code,
    #[strum(serialize = "code_interpreter_tool")]
    code_interpreter_tool,
    #[strum(serialize = "acode_interpreter_tool")]
    acode_interpreter_tool,
    #[strum(serialize = "acancel_fine_tuning_job")]
    acancel_fine_tuning_job,
    #[strum(serialize = "cancel_fine_tuning_job")]
    cancel_fine_tuning_job,
    #[strum(serialize = "alist_fine_tuning_jobs")]
    alist_fine_tuning_jobs,
    #[strum(serialize = "list_fine_tuning_jobs")]
    list_fine_tuning_jobs,
    #[strum(serialize = "aretrieve_fine_tuning_job")]
    aretrieve_fine_tuning_job,
    #[strum(serialize = "retrieve_fine_tuning_job")]
    retrieve_fine_tuning_job,
    #[strum(serialize = "responses")]
    responses,
    #[strum(serialize = "aresponses")]
    aresponses,
    #[strum(serialize = "alist_input_items")]
    alist_input_items,
    #[strum(serialize = "llm_passthrough_route")]
    llm_passthrough_route,
    #[strum(serialize = "allm_passthrough_route")]
    allm_passthrough_route,
    #[strum(serialize = "generate_content")]
    generate_content,
    #[strum(serialize = "agenerate_content")]
    agenerate_content,
    #[strum(serialize = "generate_content_stream")]
    generate_content_stream,
    #[strum(serialize = "agenerate_content_stream")]
    agenerate_content_stream,
    #[strum(serialize = "ocr")]
    ocr,
    #[strum(serialize = "aocr")]
    aocr,
    #[strum(serialize = "call_mcp_tool")]
    call_mcp_tool,
    #[strum(serialize = "list_mcp_tools")]
    list_mcp_tools,
    #[strum(serialize = "asend_message")]
    asend_message,
    #[strum(serialize = "send_message")]
    send_message,
    #[strum(serialize = "acreate_skill")]
    acreate_skill,
}

impl CallTypes {
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum PassthroughCallTypes {
    #[strum(serialize = "passthrough-image-generation")]
    passthrough_image_generation,
}
