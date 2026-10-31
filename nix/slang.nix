{
  lib,
  stdenv,
  fetchzip,
  autoPatchelfHook,
  libx11,
  zlib,
}:

let
  releases = {
    x86_64-linux = {
      arch = "x86_64";
      hash = "sha256-OFkvUUR5bS6D/8UzGHp4D170HyR+Czwy6euBEPQApfQ=";
    };
    aarch64-linux = {
      arch = "aarch64";
      hash = "sha256-4GA63RujomTuPiTuAa9OeKjPM7JDNVp2NhUzGzgcp1Q=";
    };
  };
  release = releases.${stdenv.hostPlatform.system};
in
stdenv.mkDerivation (finalAttrs: {
  pname = "spirula-slang";
  version = "2026.12.0.1";

  src = fetchzip {
    url = "https://github.com/shader-slang/slang/releases/download/v${finalAttrs.version}/slang-${finalAttrs.version}-linux-${release.arch}.tar.gz";
    inherit (release) hash;
    stripRoot = false;
  };

  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [
    stdenv.cc.cc.lib
    libx11
    zlib
  ];
  # Slang loads the SPIR-V optimizer plugin from its compiler library.
  appendRunpaths = [ "${placeholder "out"}/lib" ];
  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    cp -r bin lib share include "$out/"
    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    test "$("$out/bin/slangc" -v 2>&1)" = "${finalAttrs.version}"
    cat > smoke.slang <<'EOF'
    [shader("compute")]
    [numthreads(1, 1, 1)]
    void main() {}
    EOF
    "$out/bin/slangc" smoke.slang -entry main -target spirv -O3 -o smoke.spv
    test -s smoke.spv
    runHook postInstallCheck
  '';

  meta = {
    description = "Pinned Slang shader compiler for Spirula Studio";
    homepage = "https://shader-slang.org/";
    license = lib.licenses.asl20;
    platforms = builtins.attrNames releases;
    mainProgram = "slangc";
  };
})
