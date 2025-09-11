# Enable clang-tidy by default
option(DO_CLANGTIDY "Enable clang-tidy" OFF)

# Default executable for clang-tidy
if(NOT DEFINED CLANGTIDY_CMD)
    find_program(CLANGTIDY_CMD NAMES clang-tidy)
endif()

function(add_clangtidy_target TARGET)
    if(NOT DO_CLANGTIDY)
        return()
    endif()

    if(CLANGTIDY_CMD)
        set_target_properties(
            ${TARGET}
            PROPERTIES
                CXX_CLANG_TIDY
                "${CLANGTIDY_CMD};--checks=bugprone-*,clang-analyzer-*,cppcoreguidelines-*,modernize-*,performance-*,readability-*,-modernize-use-trailing-return-type,-performance-avoid-endl"
        )
    else()
        message(FATAL_ERROR "clang-tidy requested but executable not found")
    endif()
endfunction()
