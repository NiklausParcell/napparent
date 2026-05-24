//! Allow linking the cdylib without libpython (matches maturin / extension-module on macOS).

fn main() {
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
