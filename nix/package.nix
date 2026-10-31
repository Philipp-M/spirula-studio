{
  lib,
  stdenv,
  callPackage,
  fetchzip,
  cmake,
  ninja,
  pkg-config,
  gitMinimal,
  makeWrapper,
  copyDesktopItems,
  wayland-scanner,
  wayland,
  wayland-protocols,
  libxkbcommon,
  libdecor,
  libffi,
  libx11,
  libxcursor,
  libxrandr,
  libxinerama,
  libxi,
  libxext,
  libGL,
  vulkan-headers,
  vulkan-loader,
  curl,
  cacert,
  ffmpeg,
  xdg-utils,
  zenity,
  makeDesktopItem,
}:

let
  slang = callPackage ./slang.nix { };
  glfw = fetchzip {
    url = "https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz";
    hash = "sha256-FcnQPDeNHgov1Z07gjFze0VMz2diOrpbKZCsI96ngz0=";
  };
  imgui = fetchzip {
    url = "https://github.com/ocornut/imgui/archive/refs/tags/v1.92.8.tar.gz";
    hash = "sha256-NaSgDE5QEEMrsgYOCAPx5d0XvCIQ9T+ciAKfLFhlmzw=";
  };
  runtimeLibraries = [
    libGL
    vulkan-loader
    wayland
    libxkbcommon
    libdecor
    libx11
    libxcursor
    libxrandr
    libxinerama
    libxi
    libxext
  ];
in
stdenv.mkDerivation {
  pname = "spirula-studio";
  version = lib.head (
    builtins.match ".*set\\(SS_VERSION \"([^\"]+)\"\\).*" (builtins.readFile ../cmake/SsOptions.cmake)
  );

  src = lib.cleanSourceWith {
    src = ../.;
    filter =
      path: type:
      lib.cleanSourceFilter path type
      && !(
        type == "directory"
        && (
          lib.hasPrefix "build_" (baseNameOf path)
          || lib.hasPrefix "build-" (baseNameOf path)
          || builtins.elem (baseNameOf path) [
            "build"
            "outputs"
            "parity_refs"
          ]
        )
      );
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    gitMinimal
    makeWrapper
    copyDesktopItems
    wayland-scanner
    slang
  ];
  buildInputs = runtimeLibraries ++ [
    wayland-protocols
    vulkan-headers
    libffi
  ];

  dontUseCmakeConfigure = true;
  dontUseNinjaBuild = true;
  dontUseNinjaInstall = true;

  postPatch = ''
    # Keep the dev script's RAM limit and also respect Nix's job allocation.
    substituteInPlace build_develop.bash \
      --replace-fail 'CPU_CORES=$(nproc)' 'CPU_CORES=''${NIX_BUILD_CORES:-1}' \
      --replace-fail '--verbose -j' '--target spirula --verbose -j'

    # Generated desktop entries point at the executable behind the runtime wrapper.
    substituteInPlace src/app/gui/GuiMain.cpp \
      --replace-fail '// Before the window maps: that is when the shell looks up its entry.' "" \
      --replace-fail 'gui::register_desktop_entry();' ""

    # The dev script's source checks use Git even for an unpacked source archive.
    git init -q
    git add .
    git -c user.name=Nix -c user.email=nix@localhost \
      -c commit.gpgsign=false commit -qm 'Source snapshot'
  '';

  buildPhase = ''
    runHook preBuild
    bash build_develop.bash \
      -DSS_BACKEND=vulkan \
      -DSS_BUILD_GUI=ON \
      -DSS_BUILD_SFM=ON \
      -DSS_BUILD_SAM=ON \
      -DSS_ENABLE_PATENTED=OFF \
      -DSS_SLANGC=${slang}/bin/slangc \
      -DCMAKE_DISABLE_FIND_PACKAGE_Git=ON \
      -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
      -DFETCHCONTENT_SOURCE_DIR_GLFW=${glfw} \
      -DFETCHCONTENT_SOURCE_DIR_IMGUI=${imgui}
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -Dm755 build_vulkan/spirula "$out/bin/spirula"
    install -Dm644 assets/icon_128.png "$out/share/icons/hicolor/128x128/apps/spirula.png"
    install -Dm644 LICENSE "$out/share/licenses/spirula-studio/LICENSE"
    runHook postInstall
  '';

  postFixup = ''
    wrapProgram "$out/bin/spirula" \
      --prefix PATH : ${
        lib.makeBinPath [
          curl
          ffmpeg
          xdg-utils
          zenity
        ]
      } \
      --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath runtimeLibraries} \
      --set-default SSL_CERT_FILE ${cacert}/etc/ssl/certs/ca-bundle.crt
  '';

  desktopItems = [
    (makeDesktopItem {
      name = "spirula-studio";
      desktopName = "Spirula Studio";
      comment = "Train and edit 3D Gaussian Splatting models";
      exec = "spirula %F";
      icon = "spirula";
      startupWMClass = "spirula-studio";
      categories = [
        "Graphics"
        "3DGraphics"
      ];
      terminal = false;
    })
  ];

  meta = {
    description = "Native 3D Gaussian Splatting trainer, viewer and reconstruction tools";
    homepage = "https://github.com/harry7557558/spirula-studio";
    license = lib.licenses.gpl3Only;
    platforms = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    mainProgram = "spirula";
  };
}
