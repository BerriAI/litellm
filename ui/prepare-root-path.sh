#!/bin/sh
set -eu

source_dir=$1
runtime_dir=$2
source_config=$3
runtime_config=$4
prefix=${5%/}

case "$prefix" in
    ""|/*) ;;
    *) echo "SERVER_ROOT_PATH must start with /" >&2; exit 1 ;;
esac
case "$prefix" in
    *[!a-zA-Z0-9_./~-]*|*//*|*/./*|*/../*|*/.|*/..|*/)
        echo "SERVER_ROOT_PATH must contain safe path segments without traversal" >&2
        exit 1
        ;;
esac

mkdir -p "$runtime_dir"
cp -R "$source_dir/." "$runtime_dir/"
chmod -R u+rwX,go+rX "$runtime_dir"

if [ -n "$prefix" ]; then
    find "$runtime_dir" -type f \( -name '*.html' -o -name '*.js' -o -name '*.css' -o -name '*.txt' -o -name '*.json' \) \
        -exec sh -eu -c '
            prefix=$1
            shift
            for file do
                sed -e "s|/litellm-asset-prefix|$prefix|g" \
                    -e "s|/litellm/\.well-known/litellm-ui-config|$prefix/.well-known/litellm-ui-config|g" \
                    "$file" > "$file.tmp"
                case "$file" in
                    *.html)
                        sed -e "s|<head>|<head><meta name=\"litellm-server-root-path\" content=\"$prefix\">|g" \
                            -e "s|src=\"/ui/assets/|src=\"$prefix/ui/assets/|g" \
                            -e "s|href=\"/get_favicon\"|href=\"$prefix/get_favicon\"|g" \
                            -e "s|href=\"/favicon.ico|href=\"$prefix/favicon.ico|g" \
                            "$file.tmp" > "$file"
                        rm "$file.tmp"
                        ;;
                    *) mv "$file.tmp" "$file" ;;
                esac
            done
        ' sh "$prefix" {} +
fi

escaped_runtime=$(printf '%s' "$runtime_dir" | sed 's/[\\&|]/\\&/g')
escaped_prefix=$(printf '%s' "$prefix" | sed 's/\./\\\\./g')
sed "s|/usr/share/nginx/html|$escaped_runtime|g" "$source_config" > "$runtime_config"
if [ -n "$prefix" ]; then
    sed "/server_name _;/a\\
    absolute_redirect off;\\
    rewrite ^$escaped_prefix\$ $prefix/ui/ redirect;\\
    rewrite ^$escaped_prefix(/.*)\$ \$1 last;\\
" "$runtime_config" > "$runtime_config.tmp"
    mv "$runtime_config.tmp" "$runtime_config"
fi
