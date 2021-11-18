sgr_get_model <- function(model_name = c("afhqv2"), flavour = c("r", "t"), res = c("512", "1024"),
                           dir = rappdirs::user_data_dir("styleganr"),
                          device = c("cpu", "cuda")) {
  
  model_name <- match.arg(model_name)
  flavour <- match.arg(flavour)
  res = match.arg(res)
  device <- match.arg(device)
  
  mod <- glue::glue("stylegan3-{flavour}-{model_name}-{res}x{res}-R.zip")
  file_name <- file.path(dir, mod)
  
  if(!file.exists(file_name)) {
    url <- file.path("https://github.com/rdinnager/styleganr/releases/download/libstyleganr", 
                     basename(file_name))
    if(!dir.exists(dir)) {
      dir.create(dir, recursive = TRUE)
    }
    
    download.file(url = url, destfile = file_name)
    
  } 
    
  model <- sgr_load_model(file_name, device = device)
  
  return(model)
}

sgr_load_model <- function(file_name, device = c("cpu", "cuda")) {
  
  device <- match.arg(device)
  
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  
  unzip(file_name, exdir = tmp)
  
  pt_file <- file.path(tmp, gsub(".zip", ".pt", basename(file_name)))
  json_file <- file.path(tmp, gsub(".zip", ".json", basename(file_name)))
  
  state_dict <- load_state_dict(pt_file)
  args <- jsonlite::read_json(json_file)
  
  model <- rlang::exec(Generator, !!!args)
  model$load_state_dict(state_dict)
  
  if(device == "cuda") {
    model <- model$cuda()
  }
  
  return(model)
}