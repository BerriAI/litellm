#!/bin/sh
set -eu

version=3.5.9
sha256=603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a

curl -fsSL -o openssl.tar.gz "https://github.com/openssl/openssl/releases/download/openssl-${version}/openssl-${version}.tar.gz"
echo "${sha256}  openssl.tar.gz" | sha256sum -c -
tar xzf openssl.tar.gz --strip-components=1
./Configure --prefix=/opt/prisma-openssl --openssldir=/opt/prisma-openssl/ssl shared no-tests
make -s -j"$(nproc)" build_libs
mkdir -p /runtime/usr/lib /runtime/opt/prisma-openssl/ssl
install -m 0755 libssl.so.3 libcrypto.so.3 /runtime/usr/lib/
ln -s /etc/ssl/cert.pem /etc/ssl/certs /runtime/opt/prisma-openssl/ssl/
