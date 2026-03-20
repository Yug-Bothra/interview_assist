#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let backend_dir = if cfg!(debug_assertions) {
                // DEV MODE: backend lives one folder up
                std::path::PathBuf::from("../backend")
            } else {
                // BUILD MODE: backend is copied into resources/
                app.path_resolver().resource_dir().unwrap().join("backend")
            };

            println!("Launching backend from: {:?}", backend_dir);

            std::thread::spawn(move || {
                let mut cmd = Command::new("python");

                cmd.current_dir(&backend_dir)
                    .arg("-m")
                    .arg("app.main")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null());

                match cmd.spawn() {
                    Ok(_) => println!("Backend started."),
                    Err(e) => println!("Failed to start backend: {:?}", e),
                }
            });

            // wait a bit for backend to boot
            thread::sleep(Duration::from_secs(2));

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error running tauri application");
}
