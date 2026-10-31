{
  description = "Spirula Studio — native 3D Gaussian Splatting trainer";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          spirula = pkgs.callPackage ./nix/package.nix { };
        in
        {
          default = spirula;
          inherit spirula;
        }
      );

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.spirula}/bin/spirula";
          meta.description = "Launch Spirula Studio or run a CLI subcommand";
        };
      });

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
