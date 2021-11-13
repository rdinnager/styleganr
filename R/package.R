## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @importFrom utils download.file packageDescription unzip
## usethis namespace: end
NULL

.onLoad <- function(lib, pkg) {
  if (torch::torch_is_installed()) {

    if (!styleganr_is_installed())
      install_styleganr()

    if (!styleganr_is_installed()) {
      if (interactive())
        warning("libstyleganr is not installed. Run `intall_styleganr()` before using the package.")
    } else {
      dyn.load(lib_path(), local = FALSE)

      # when using devtools::load_all() the library might be available in
      # `lib/pkg/src`
      pkgload <- file.path(lib, pkg, "src", paste0(pkg, .Platform$dynlib.ext))
      if (file.exists(pkgload))
        dyn.load(pkgload)
      else
        library.dynam("styleganr", pkg, lib)
    }
  }
}

inst_path <- function() {
  install_path <- Sys.getenv("STYLEGANR_HOME")
  if (nzchar(install_path)) return(install_path)

  system.file("deps", package = "styleganr")
}

lib_path <- function() {
  install_path <- inst_path()

  if (.Platform$OS.type == "unix") {
    file.path(install_path, "lib", paste0("libstyleganr", lib_ext()))
  } else {
    file.path(install_path, "lib", paste0("styleganr", lib_ext()))
  }
}

lib_ext <- function() {
  if (grepl("darwin", version$os))
    ".dylib"
  else if (grepl("linux", version$os))
    ".so"
  else
    ".dll"
}

styleganr_is_installed <- function() {
  file.exists(lib_path())
}

install_styleganr <- function(url = Sys.getenv("STYLEGANR_URL", unset = NA), cuda_version = Sys.getenv("CUDA")) {
  
  assertthat::assert_that(cuda_version %in% c("", "10.2", "11.1"))

  if (!interactive() && Sys.getenv("TORCH_INSTALL", unset = 0) == "0") return()

  if (is.na(url)) {
    tmp <- tempfile(fileext = ".zip")
    version <- packageDescription("styleganr")$Version
    os <- get_cmake_style_os()
    if (torch::cuda_is_available()) {
      if(cuda_version == "") {
        stop("CUDA installation of `torch` detected. You must specify which version of CUDA using the cuda_version argument or by setting the CUDA environmental variable")
      }
      dev <- paste0("cu", cuda_version)
    } else {
      dev <- "cpu"
    }

    url <- sprintf("https://github.com/rdinnager/styleganr/releases/download/libstyleganr/styleganr-%s+%s-%s.zip",
                   version, dev, os)
  }

  if (is_url(url)) {
    file <- tempfile(fileext = ".zip")
    on.exit(unlink(file), add = TRUE)
    download.file(url = url, destfile = file)
  } else {
    message('Using file ', url)
    file <- url
  }

  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  unzip(file, exdir = tmp)

  file.copy(
    list.files(list.files(tmp, full.names = TRUE), full.names = TRUE),
    inst_path(),
    recursive = TRUE
  )
}

get_cmake_style_os <- function() {
  os <- version$os
  if (grepl("darwin", os)) {
    "Darwin"
  } else if (grepl("linux", os)) {
    "Linux"
  } else {
    "win64"
  }
}

is_url <- function(x) {
  grepl("^https", x) || grepl("^http", x)
}

