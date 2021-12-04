if (!require(fs, quietly = TRUE)) {
  install.packages("fs")
}

if (!require(fs))
  stop("fs was not correctly installed?")

if (dir.exists("csrc")) {
  cat("Building styleganr torch plugins .... \n")
  
  if (!fs::dir_exists("csrc/build"))
    fs::dir_create("csrc/build")
  
  withr::with_dir("csrc/build", {
    system("cmake ..")
    output <- system("cmake --build . --target package --config Release --parallel 8", intern = TRUE)
    message("copying libraries ...")
    system("cmake --install .")
  })
  
  # # copy lantern
  # source("R/lantern_sync.R")
  # lantern_sync(TRUE)  
  # 
  # # download torch
  # source("R/install.R")
  # install_torch(path = normalizePath("deps/"), load = FALSE)
  # 
  # # copy deps to inst
  # if (fs::dir_exists("inst/deps"))
  #   fs::dir_delete("inst/deps/")
  # 
  # fs::dir_copy("deps/", new_path = "inst/deps/")
}


