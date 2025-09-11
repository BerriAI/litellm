# Only add this project if Infer is not defined
if(TARGET Infer)
    return()
endif()

include(ExternalProject)
# Only grab infer dependencies if we need to
if(DO_INFER)
    if(NOT Infer_EXECUTABLE OR NOT EXISTS "${Infer_EXECUTABLE}")
        set(TAG_INFER
            "v1.1.0"
            CACHE STRING "infer github tag")

        set(Infer_BUILD_DIR ${CMAKE_BINARY_DIR}/infer)
        set(Infer_ROOT ${Infer_BUILD_DIR}/infer-${TAG_LIBDATADOG})

        message(STATUS "${CMAKE_CURRENT_LIST_DIR}/tools/fetch_infer.sh ${TAG_INFER} ${Infer_ROOT}")
        execute_process(COMMAND "${CMAKE_CURRENT_LIST_DIR}/tools/fetch_infer.sh" ${TAG_INFER} ${Infer_ROOT}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} COMMAND_ERROR_IS_FATAL ANY)

        set(Infer_DIR "${Infer_ROOT}")
        set(Infer_EXECUTABLE "${Infer_DIR}/bin/infer")
    endif()
endif()

# Add a target for using infer.  It does nothing if DO_INFER is not set
function(add_infer_target TARGET)
    # Automatically generate the infer target name
    set(NAME "infer_dd_${TARGET}")

    if(NOT TARGET ${TARGET})
        message(FATAL_ERROR "Target ${TARGET} does not exist")
        return()
    endif()

    if(DO_INFER)
        # Initialize command variable
        set(infer_cmd ${Infer_EXECUTABLE} --compilation-database ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json)

        # Define the custom target with the constructed command
        add_custom_target(
            ${NAME}
            COMMAND ${infer_cmd}
            COMMENT "Running infer on ${TARGET}")

        # infer can't seem to find stdlib headers, so we have to add them to the target so they are included in the
        # compile_commands.json
        foreach(inc ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
            target_include_directories(${TARGET} PRIVATE ${inc})
        endforeach()

        # Make the infer target a dependent of the specified target
        add_dependencies(${TARGET} ${NAME})
    else()
        # Define a do-nothing target if infer is disabled
        add_custom_target(
            ${NAME}
            COMMAND echo "infer target ${NAME} is disabled."
            COMMENT "infer is disabled for ${TARGET}")
    endif()
endfunction()
