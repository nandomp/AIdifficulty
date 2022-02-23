# -----------------------------------------------------------------------------------------------------
# Packages
# -----------------------------------------------------------------------------------------------------

options( java.parameters = "-Xmx12g" )
.lib<- c("mirt", "ggplot2", "reshape2", "gridExtra", "tidyverse", "ggplot2", "randomForest", "purrr", "caret", "skimr", "DT", "reshape2", "doParallel", "farff", "ggthemes")

.inst <- .lib %in% installed.packages()
if (length(.lib[!.inst])>0) install.packages(.lib[!.inst], repos=c("http://rstudio.org/_packages", "http://cran.rstudio.com")) 
lapply(.lib, require, character.only=TRUE)

pal = colorRampPalette(c(hcl(0,100, c(20,100)), hcl(240,100,c(100,20))))

# -----------------------------------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------------------------------

openPDFEPS <- function(file, height= PDFheight, width= PDFwidth, PDFEPS = 1) {
  if (PDFEPS == 1) {
    pdf(paste(file, ".pdf", sep=""), width, height)
  } else if (PDFEPS == 2) {
    postscript(paste(file, ".eps", sep=""), width, height, horizontal=FALSE)
  }
}

formatData4IRT <- function(dat.orig){
  
  dat <- dat.orig
  
  nrow(unique(dat))
  nrow(dat)
  dat <- unique(dat)
  ncol(dat)
  mod.names <- dat[,1]
  dat <- dat[,-1]
  ncol(dat)
  if(length(unique(dat[, ncol(dat)]))>2){
    dat <- dat[, -ncol(dat)]  
  }
  
  # colnames(dat) <- seq(1:ncol(dat))
  # row.names(dat) <- mod.names
  
  # rownames(dat) <- dat.orig[,1]
  ncol(dat)
  colnames(dat)[ncol(dat)]
  
  whichNO <- which(apply(dat, 2, var, na.rm=TRUE) == 0)
  # dat.c <- dat[,apply(dat, 2, var, na.rm=TRUE) != 0] #remove images with variability == 0
  
  if(length(whichNO)>0){
    dat.c <- dat[,-whichNO] #remove images with variability == 0
    
  }else{
    dat.c <- dat
  }
  
  
  
  ncol(dat.c)
  # sapply(dat.c, class)
  # class(dat.c)
  dat.c <- as.data.frame(dat.c)
  # saveRDS(dat.c, file = paste0(path,"/", title,".RDS"))
  return(list(dat.c, whichNO))
}

modelIRT <- function(dat.irt, method = "2PL"){
  
  
  # mirtCluster() 
  if(method == "2PL"){
    # set.seed(1)
    fit <- mirt(dat.irt, 1, itemtype = '2PL', technical = list(NCYCLES = 300))
    
  }else{
    if(method == "1PL"){
      
    }else{
      fit <- mirt(dat.irt, 1, itemtype = '1PL', technical = list(NCYCLES = 300))
    }
    # set.seed(1)
  }
  
  temp = coef(fit, simplify = T, IRTpars =T)$items
  params <- data.frame(temp[,c("g","b","a")])
  colnames(params)<-c("Gussng","Dffclt","Dscrmn")
  plot(fit)
  
  # computing the abilities 'ab_vector' of the respondents   
  abil<- as.data.frame(t(fscores(fit)))
  colnames(abil) = rownames(dat.irt) 
  rownames(abil) = NULL
  return(list(fit, params, abil))
}



# -----------------------------------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------------------------------

# dat.orig <- "results_Letters_174x20000.rds"
# orig.dataset <- read.csv("letter.csv")
# splitcols4IRT <- 40
# Diff.filter <- 3
# # dat.orig %>% head(); nrow(dat.orig)

ETL.ITR <- function(dat.orig, orig.dataset, splitcols4IRT, nameDS = ""){
  # Format dat.orig into IRT format (no variability removal, etc.)
  temp <- formatData4IRT(dat.orig)
  dat.irt <- temp[[1]]
  # nrow(dat.irt); ncol(dat.irt)
  whichRemoved <- temp[[2]]
  # length(whichRemoved)
  
  splits = splitcols4IRT
  
  if (splits > 1){
    num.samples = ceiling(ncol(dat.irt)/splits)
  }else{
    num.samples = ncol(dat.irt)
  }
  
  all.params <- data.frame(Gussng = NA, Dffclt = NA, Dscrmn = NA)
  
  # Parallelisation -------------------------
  # no_cores <- detectCores()/2 
  # cl <- makePSOCKcluster(no_cores)
  # registerDoParallel(cores=cl)  
  # -----------------------------------------
  
  for (i in 1:splits){
    print(paste0("SAMPLE:  ", (1+(num.samples*(i-1))), " - ", num.samples*i, " ------"))
    max = num.samples*i
    if((num.samples*i)>ncol(dat.irt)){
      max = ncol(dat.irt)
    }
    
    res.sample <- dat.irt[,(1+(num.samples*(i-1))):(max)]
    set.seed(1)
    fit <- mirt(res.sample, 1, itemtype = 'Rasch', technical = list(NCYCLES = 200))
    
    # openPDFEPS(paste0("Fits/FitIRT_",i),width=7,height=7)
    # print(plot(fit))
    # dev.off()
    
    temp = coef(fit, simplify = T, IRTpars =T)$items
    params <- data.frame(temp[,c("g","b","a")])
    colnames(params)<-c("Gussng","Dffclt","Dscrmn")
    if(i == 1){
      all.params <- rbind(all.params[0,], params)
    }else{
      all.params <- rbind(all.params, params)
    }
    
  }
  
  # Parallelisation -------------------------
  # stopCluster(cl)
  # registerDoSEQ()
  # -----------------------------------------
  
  
  # all.params <- all.params[-1,]
  # saveRDS(all.params, file = paste0(nameDS,"_allIRTparams.RDS"))
  # all.params <- readRDS("tempIRTparams.RDS")
  all.params <- all.params[,-1]
  nrow(all.params)
  nrow(orig.dataset)


  meanACC <- as.vector(colMeans(dat.orig[,-1]))
  if(length(unique(whichRemoved))>0){
    orig.dataset <- orig.dataset[-as.numeric(whichRemoved),]
    meanACC <- meanACC[-as.numeric(whichRemoved)]
    
  }
  nrow(orig.dataset)
  orig.dataset$meanACC <- meanACC
  
  # for(i in whichRemoved){
  #   print(i)
  #   print(nrow(orig.dataset))
  #   orig.dataset <- orig.dataset[-as.numeric(i),]
  #   print(nrow(orig.dataset))
  #   print("..................")
  #   
  # }
  
  
  
  # orig.irt <- orig.dataset[,-ncol(orig.dataset)]
  
  orig.irt <- orig.dataset
  print(paste0("nRow IRT data (params): ", nrow(all.params), " - nRow dataset: ", nrow(orig.irt)))
  orig.irt$Dffclt <- all.params$Dffclt
  orig.irt$Dscrmn <- all.params$Dscrmn
  # orig.irt <- orig.irt[, -(ncol(orig.irt)-2)]
  
  # 
  # orig.irt <- cbind(orig.irt, all.params$Dffclt, all.params$Dscrmn)
  # orig.irt <- orig.irt[, -(ncol(orig.irt)-2)]
  # colnames(orig.irt)[ncol(orig.irt-2)] <- "Dffclt"
  # colnames(orig.irt)[ncol(orig.irt-1)] <- "Dscrmn"
  
  return(orig.irt)
}

croppDiffs <- function(orig.irt, Diff.filter, cropp = F){
  
  if(cropp){
    orig.irt.f <- filter(orig.irt, Dffclt < Diff.filter, Dffclt > - Diff.filter)
    
  }else{
    orig.irt[which(orig.irt$Dffclt > Diff.filter),"Dffclt"] <- Diff.filter
    orig.irt[which(orig.irt$Dffclt < - Diff.filter),"Dffclt"] <- -Diff.filter
    orig.irt.f <- orig.irt
  }
  
  return(orig.irt.f)
}

croppDiffs.list <- function(orig.irt, Diff.filter, cropp = F){
  
  if(cropp){
    orig.irt.f <- filter(orig.irt, Dffclt < Diff.filter, Dffclt > - Diff.filter)
    removed = nrow(orig.irt) - nrow(orig.irt.f)
    
  }else{
    orig.irt[which(orig.irt$Dffclt > Diff.filter),"Dffclt"] <- Diff.filter
    orig.irt[which(orig.irt$Dffclt < - Diff.filter),"Dffclt"] <- -Diff.filter
    orig.irt.f <- orig.irt
    removed = 0
  }
  
  return(list(orig.irt.f, removed))
}

plotDiffHistogram <- function(orig.irt, title = "", subtitle = ""){
  
  return(ggplot(orig.irt, aes(Dffclt)) + geom_histogram(bins = 100, fill = "#9bdad4", colour = "#74b7be") + theme_minimal() + 
           labs(title = title, subtitle = subtitle)) + ylab("") + xlab("Difficulty")
}



# -----------------------------------------------------------------------------------------------------
# Modelling
# -----------------------------------------------------------------------------------------------------

# Hiperparams
# -----------------------------------------------------------------------------------------------------

hiperRF <- expand.grid(mtry = c(1,2,3,4),
                       min.node.size = c(2, 3, 4, 5, 10, 15, 20, 30),
                       splitrule = c("variance", "extratrees"))


hiperGB <- expand.grid(shrinkage = seq(0.1, 1, by = 0.2), 
                       interaction.depth = c(1, 3, 7, 10),
                       n.minobsinnode = c(2, 5, 10, 15),
                       n.trees = c(100, 300, 500, 1000, 2000))


# hiperRF.df <- expand.grid(mtry = c(1,2,3,4),
#                        min.node.size = c(2, 3, 4, 5, 10, 15, 20, 30),
#                        splitrule = c("variance", "extratrees"))
# 
# 
# hiperGB.df <- expand.grid(shrinkage = seq(0.1, 1, by = 0.2), 
#                        interaction.depth = c(1, 3, 7, 10),
#                        n.minobsinnode = c(2, 5, 10, 15),
#                        n.trees = c(100, 300, 500, 1000, 2000))



hiperKNN <- data.frame(k = c(1, 2, 3, 5, 7, 9, 11, 15, 20, 30, 50))

hiperRPART <- expand.grid(cp = seq(0, 0.05, 0.005))

hiperCTREE = expand.grid(maxdepth = c(1,2,3,5,10), mincriterion = c(0.99,0.98, 0.97))

hiperNNET <- expand.grid(size = c(1, 3, 5, 7, 10, 20, 50, 80, 100, 120),
                         decay = c(0.0001, 0.1, 0.5))

# Training Setting
# -----------------------------------------------------------------------------------------------------

particiones  <- 5
repeticiones <- 2

createPartitions <- function(orig.irt, partition = 0.7, Class2Model){
  set.seed(288)
  train <- createDataPartition(y = orig.irt[,Class2Model], p = 0.7, list = FALSE, times = 1)
  datos_train <- orig.irt[train, ]
  datos_test  <- orig.irt[-train, ]
  return(list(datos_train, datos_test))
}

control_train <- trainControl(method = "repeatedcv", number = particiones,
                              repeats = repeticiones, #seeds = seeds,
                              returnResamp = "final", verboseIter = FALSE,
                              allowParallel = TRUE, savePredictions = 'final')

# Models
# -----------------------------------------------------------------------------------------------------

trainModels <- function(datos_train, model ="", scale = F, tuning = F){
  
  if(model == "lmStepAIC"){
    set.seed(288)
    if (scale){
      step_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                          method = "lmStepAIC", trControl = control_train, trace = FALSE, metric = "RMSE", verbose = TRUE)
    }else{
      step_model <- train(Dffclt ~ ., data = datos_train, 
                          method = "lmStepAIC", trControl = control_train, trace = FALSE, metric = "RMSE", verbose = TRUE)
    }
    return(step_model)
  }
  if(model == "ranger"){
    set.seed(288)
    if (tuning){
      if (scale){
        modelo_rf <- train(Dffclt ~ ., data = datos_train, 
                           preProcess = c("center", "scale"),
                           method = "ranger",
                           tuneGrid = hiperRF,
                           trControl = control_train,
                           # N?mero de ?rboles ajustados
                           num.trees = 1000, 
                           metric = "RMSE", verbose = F, importance = 'impurity')
      }else{
        modelo_rf <- train(Dffclt ~ ., data = datos_train,
                           # preProcess = c("center", "scale"),
                           method = "ranger",
                           tuneGrid = hiperRF,
                           trControl = control_train,
                           # N?mero de ?rboles ajustados
                           num.trees = 1000, 
                           metric = "RMSE", verbose = F, importance = 'impurity')
      }
    }else{ #no tuning 
      if (scale){
        modelo_rf <- train(Dffclt ~ ., data = datos_train,
                           preProcess = c("center", "scale"),
                           method = "ranger",
                           trControl = control_train,
                           tuneLength = 1, num.trees = 1000,
                           metric = "RMSE", verbose = F, importance = 'impurity')
      }else{
        modelo_rf <- train(Dffclt ~ ., data = datos_train,
                           # preProcess = c("center", "scale"),
                           method = "ranger",
                           trControl = control_train,
                           tuneLength = 1, num.trees = 1000,
                           metric = "RMSE", verbose = F, importance = 'impurity')
      }
    }
    
    return(modelo_rf)
    
  }
  if(model == "glmnet"){
    set.seed(288)
    if(tuning){
      if(scale){
        glmnet_model <- train(Dffclt ~ ., data = datos_train, 
                              preProcess = c("center", "scale"),
                              method = "glmnet", trControl = control_train, metric = "RMSE", verbose = F)
      }else{
        glmnet_model <- train(Dffclt ~ ., data = datos_train, 
                              # preProcess = c("center", "scale"),
                              method = "glmnet", trControl = control_train, metric = "RMSE", verbose = F)
      }
    }else{
      if(scale){
        glmnet_model <- train(Dffclt ~ ., data = datos_train, 
                              preProcess = c("center", "scale"),
                              method = "glmnet", trControl = control_train, tuneLength = 1, metric = "RMSE", verbose = F)
      }else{
        glmnet_model <- train(Dffclt ~ ., data = datos_train, 
                              # preProcess = c("center", "scale"),
                              method = "glmnet", trControl = control_train, tuneLength = 1, metric = "RMSE", verbose = F)
      }
    }
    return(glmnet_model)
    
  }
  if(model == "gbm"){
    set.seed(288)
    if(tuning){
      if(scale){
        gbm_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                           method = "gbm", trControl = control_train, tuneGrid = hiperGB, verbose = FALSE, metric = "RMSE")
      }else{
        gbm_model <- train(Dffclt ~ ., data = datos_train, 
                           method = "gbm", trControl = control_train, tuneGrid = hiperGB, verbose = FALSE, metric = "RMSE")
      }
    }else{# no tuning
      if(scale){
        gbm_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                           method = "gbm", trControl = control_train, tuneLength = 1, verbose = FALSE, metric = "RMSE")
      }else{
        gbm_model <- train(Dffclt ~ ., data = datos_train, 
                           method = "gbm", trControl = control_train, tuneLength = 1, verbose = FALSE, metric = "RMSE")
      }
    }
    return(gbm_model)
  }
  if(model == "knn"){
    set.seed(288)
    if(tuning){
      if(scale){
        knn_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                           method = "knn", trControl = control_train, tuneGrid = hiperKNN, verbose = FALSE, metric = "RMSE", verbose = F)
      }else{
        knn_model <- train(Dffclt ~ ., data = datos_train, 
                           method = "knn", trControl = control_train, tuneGrid = hiperKNN, verbose = FALSE, metric = "RMSE", verbose = F)
      }
    }else{
      if(scale){
        knn_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                           method = "knn", trControl = control_train, tuneLength = 1, metric = "RMSE", verbose = F)
      }else{
        knn_model <- train(Dffclt ~ ., data = datos_train, 
                           method = "knn", trControl = control_train, tuneLength = 1, metric = "RMSE", verbose = F)
      }
    }
    
    return(knn_model)
  }
  if(model == "rpart"){
    set.seed(288)
    if(tuning){
      if(scale){
        rpart_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                             method = "rpart", trControl = control_train, tuneGrid = hiperRPART, metric = "RMSE", verbose = TRUE)
      }else{
        rpart_model <- train(Dffclt ~ ., data = datos_train, 
                             method = "rpart", trControl = control_train, tuneGrid = hiperRPART, metric = "RMSE", verbose = TRUE)
      }
    }else{
      if(scale){
        rpart_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                             method = "rpart", trControl = control_train, metric = "RMSE", verbose = F)
      }else{
        rpart_model <- train(Dffclt ~ ., data = datos_train, 
                             method = "rpart", trControl = control_train, metric = "RMSE", verbose = F)
      }
    }
    
    return(rpart_model)
  }
  
  # if(model == "ctree"){
  #   set.seed(288)
  #   if(scale){
  #     ctree_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
  #                          method = "ctree2", trControl = control_train, tuneGrid = hiperCTREE, metric = "RMSE")
  #   }else{
  #     ctree_model <- train(Dffclt ~ ., data = datos_train, 
  #                          method = "ctree2", trControl = control_train, tuneGrid = hiperCTREE, metric = "RMSE")
  #   }
  #   return(ctree_model)
  # }
  
  if(model == "nnet"){
    set.seed(288)
    if(tuning){
      if(scale){
        NNET_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                            method = "nnet", trControl = control_train, tuneGrid = hiperNNET, metric = "RMSE",  
                            # Rango de inicializaci?n de los pesos
                            rang = c(-0.7, 0.7),
                            # N?mero m?ximo de pesos
                            # se aumenta para poder incluir m?s meuronas
                            MaxNWts = 2000,
                            # Para que no se muestre cada iteraci?n por pantalla
                            trace = T)
      }else{
        NNET_model <- train(Dffclt ~ ., data = datos_train, 
                            method = "nnet", trControl = control_train, tuneGrid = hiperNNET, metric = "RMSE",
                            # Rango de inicializaci?n de los pesos
                            rang = c(-0.7, 0.7),
                            # N?mero m?ximo de pesos
                            # se aumenta para poder incluir m?s meuronas
                            MaxNWts = 2000,
                            # Para que no se muestre cada iteraci?n por pantalla
                            trace = T)
      }
    }else{
      if(scale){
        NNET_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                            method = "nnet", trControl = control_train, metric = "RMSE",  
                            # Rango de inicializaci?n de los pesos
                            rang = c(-0.7, 0.7),
                            # N?mero m?ximo de pesos
                            # se aumenta para poder incluir m?s meuronas
                            MaxNWts = 2000,
                            # Para que no se muestre cada iteraci?n por pantalla
                            trace = T)
      }else{
        NNET_model <- train(Dffclt ~ ., data = datos_train, 
                            method = "nnet", trControl = control_train, metric = "RMSE",
                            # Rango de inicializaci?n de los pesos
                            rang = c(-0.7, 0.7),
                            # N?mero m?ximo de pesos
                            # se aumenta para poder incluir m?s meuronas
                            MaxNWts = 2000,
                            # Para que no se muestre cada iteraci?n por pantalla
                            trace = T)
      }
    }
    
    return(NNET_model)
  }
  
  
  
  
  
}


trainModels.Perf <- function(datos_train, model ="", scale = F, tuning = F){
  
  if(model == "lmStepAIC"){
    set.seed(288)
    if (scale){
      step_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                          method = "lmStepAIC", trControl = control_train, trace = FALSE, verbose = TRUE)
    }else{
      step_model <- train(Dffclt ~ ., data = datos_train, 
                          method = "lmStepAIC", trControl = control_train, trace = FALSE, verbose = TRUE)
    }
    return(step_model)
  }
  if(model == "ranger"){
    set.seed(288)
    if (tuning){
      if (scale){
        modelo_rf <- train(Dffclt ~ ., data = datos_train, 
                           preProcess = c("center", "scale"),
                           method = "ranger",
                           tuneGrid = hiperRF,
                           trControl = control_train,
                           # N?mero de ?rboles ajustados
                           
                           verbose = F)
      }else{
        modelo_rf <- train(Dffclt ~ ., data = datos_train,
                           # preProcess = c("center", "scale"),
                           method = "ranger",
                           tuneGrid = hiperRF,
                           trControl = control_train,
                           # N?mero de ?rboles ajustados
                           
                            verbose = F)
      }
    }else{ #no tuning 
      if (scale){
        modelo_rf <- train(Dffclt ~ ., data = datos_train,
                           preProcess = c("center", "scale"),
                           method = "ranger",
                           trControl = control_train,
                           tuneLength = 1, 
                           verbose = F)
      }else{
        modelo_rf <- train(Dffclt ~ ., data = datos_train,
                           # preProcess = c("center", "scale"),
                           method = "ranger",
                           trControl = control_train,
                           tuneLength = 1, 
                           verbose = F)
      }
    }
    
    return(modelo_rf)
    
  }
  if(model == "glmnet"){
    set.seed(288)
    if(tuning){
      if(scale){
        glmnet_model <- train(Dffclt ~ ., data = datos_train, 
                              preProcess = c("center", "scale"),
                              method = "glmnet", trControl = control_train, verbose = F)
      }else{
        glmnet_model <- train(Dffclt ~ ., data = datos_train, 
                              # preProcess = c("center", "scale"),
                              method = "glmnet", trControl = control_train, verbose = F)
      }
    }else{
      if(scale){
        glmnet_model <- train(Dffclt ~ ., data = datos_train, 
                              preProcess = c("center", "scale"),
                              method = "glmnet", trControl = control_train, tuneLength = 1, verbose = F)
      }else{
        glmnet_model <- train(Dffclt ~ ., data = datos_train, 
                              # preProcess = c("center", "scale"),
                              method = "glmnet", trControl = control_train, tuneLength = 1, verbose = F)
      }
    }
    return(glmnet_model)
    
  }
  if(model == "gbm"){
    set.seed(288)
    if(tuning){
      if(scale){
        gbm_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                           method = "gbm", trControl = control_train, tuneGrid = hiperGB, verbose = FALSE)
      }else{
        gbm_model <- train(Dffclt ~ ., data = datos_train, 
                           method = "gbm", trControl = control_train, tuneGrid = hiperGB, verbose = FALSE)
      }
    }else{# no tuning
      if(scale){
        gbm_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                           method = "gbm", trControl = control_train, tuneLength = 1, verbose = FALSE)
      }else{
        gbm_model <- train(Dffclt ~ ., data = datos_train, 
                           method = "gbm", trControl = control_train, tuneLength = 1, verbose = FALSE)
      }
    }
    return(gbm_model)
  }
  if(model == "knn"){
    set.seed(288)
    if(tuning){
      if(scale){
        knn_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                           method = "knn", trControl = control_train, tuneGrid = hiperKNN, verbose = FALSE,  verbose = F)
      }else{
        knn_model <- train(Dffclt ~ ., data = datos_train, 
                           method = "knn", trControl = control_train, tuneGrid = hiperKNN, verbose = FALSE,  verbose = F)
      }
    }else{
      if(scale){
        knn_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                           method = "knn", trControl = control_train, tuneLength = 1, verbose = F)
      }else{
        knn_model <- train(Dffclt ~ ., data = datos_train, 
                           method = "knn", trControl = control_train, tuneLength = 1, verbose = F)
      }
    }
    
    return(knn_model)
  }
  if(model == "rpart"){
    set.seed(288)
    if(tuning){
      if(scale){
        rpart_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                             method = "rpart", trControl = control_train, tuneGrid = hiperRPART, metric = "RMSE", verbose = TRUE)
      }else{
        rpart_model <- train(Dffclt ~ ., data = datos_train, 
                             method = "rpart", trControl = control_train, tuneGrid = hiperRPART, metric = "RMSE", verbose = TRUE)
      }
    }else{
      if(scale){
        rpart_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                             method = "rpart", trControl = control_train, metric = "RMSE", verbose = F)
      }else{
        rpart_model <- train(Dffclt ~ ., data = datos_train, 
                             method = "rpart", trControl = control_train, metric = "RMSE", verbose = F)
      }
    }
    
    return(rpart_model)
  }
  
  # if(model == "ctree"){
  #   set.seed(288)
  #   if(scale){
  #     ctree_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
  #                          method = "ctree2", trControl = control_train, tuneGrid = hiperCTREE, metric = "RMSE")
  #   }else{
  #     ctree_model <- train(Dffclt ~ ., data = datos_train, 
  #                          method = "ctree2", trControl = control_train, tuneGrid = hiperCTREE, metric = "RMSE")
  #   }
  #   return(ctree_model)
  # }
  
  if(model == "nnet"){
    set.seed(288)
    if(tuning){
      if(scale){
        NNET_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                            method = "nnet", trControl = control_train, tuneGrid = hiperNNET, metric = "RMSE",  
                            # Rango de inicializaci?n de los pesos
                            rang = c(-0.7, 0.7),
                            # N?mero m?ximo de pesos
                            # se aumenta para poder incluir m?s meuronas
                            MaxNWts = 2000,
                            # Para que no se muestre cada iteraci?n por pantalla
                            trace = T)
      }else{
        NNET_model <- train(Dffclt ~ ., data = datos_train, 
                            method = "nnet", trControl = control_train, tuneGrid = hiperNNET, metric = "RMSE",
                            # Rango de inicializaci?n de los pesos
                            rang = c(-0.7, 0.7),
                            # N?mero m?ximo de pesos
                            # se aumenta para poder incluir m?s meuronas
                            MaxNWts = 2000,
                            # Para que no se muestre cada iteraci?n por pantalla
                            trace = T)
      }
    }else{
      if(scale){
        NNET_model <- train(Dffclt ~ ., data = datos_train, preProcess = c("center", "scale"),
                            method = "nnet", trControl = control_train, metric = "RMSE",  
                            # Rango de inicializaci?n de los pesos
                            rang = c(-0.7, 0.7),
                            # N?mero m?ximo de pesos
                            # se aumenta para poder incluir m?s meuronas
                            MaxNWts = 2000,
                            # Para que no se muestre cada iteraci?n por pantalla
                            trace = T)
      }else{
        NNET_model <- train(Dffclt ~ ., data = datos_train, 
                            method = "nnet", trControl = control_train, metric = "RMSE",
                            # Rango de inicializaci?n de los pesos
                            rang = c(-0.7, 0.7),
                            # N?mero m?ximo de pesos
                            # se aumenta para poder incluir m?s meuronas
                            MaxNWts = 2000,
                            # Para que no se muestre cada iteraci?n por pantalla
                            trace = T)
      }
    }
    
    return(NNET_model)
  }
  
  
  
  
  
}

# -----------------------------------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------------------------------

# Training
# -----------------------------------------------------------------------------------------------------

modelResampling <- function(models, sd.data = 1){
  
  mods <- resamples(models)
  # summary(mods)
  # mods$values
  
  metrics_resample <- mods$values %>%
    gather(key = "model", value = "value", -Resample) %>%
    separate(col = "model", into = c("model", "metric"),
             sep = "~", remove = TRUE)
  # metrics_resample %>% head()
  
  # normalised RMSE data 
  
  
  temp <- metrics_resample %>% filter(metric == "RMSE") %>%
    group_by(Resample, model) %>% 
    summarise(metric = "NRMSE", value = value/sd.data) 
  
  metrics_resample <- rbind(metrics_resample, temp)
  
  
  metrics_resample %>% 
    group_by(model, metric) %>% 
    summarise(mean = mean(value)) %>%
    spread(key = metric, value = mean) %>%
    arrange(desc(RMSE))
  
  return(metrics_resample)
}

summaryResampling <- function(metrics_resample, ds = ""){
  
  mean.metrics <- metrics_resample %>%
    group_by(model, metric) %>% 
    summarise(mean = mean(value)) %>% 
    dcast(model ~ metric)
  
  sd.metrics <- metrics_resample %>%
    group_by(model, metric) %>% 
    summarise(sd = sd(value)) %>% 
    dcast(model ~ metric)
  
  
  return(data.frame(Dataset = ds,
                    Model = mean.metrics$model, 
                    MAE.mean = mean.metrics$MAE, MAE.sd = sd.metrics$MAE,
                    RMSE.mean = mean.metrics$RMSE, RMSE.sd =sd.metrics$RMSE,
                    NRMSE.mean = mean.metrics$NRMSE, NRMSE.sd =sd.metrics$NRMSE,
                    Rsquared.mean = mean.metrics$Rsquared, Rsquared.sd =sd.metrics$Rsquared
  ))
  
}

plotResampling <- function(metrics_resample, title = ""){
  byRMSE <- metrics_resample %>% filter(metric == "RMSE") %>%
    group_by(model) %>% 
    summarise(mean = mean(value)) %>%
    ggplot(aes(x = reorder(model, mean), y = mean, label = round(mean, 2))) +
    geom_segment(aes(x = reorder(model, mean), y = 0,
                     xend = model, yend = mean),
                 color = "grey50") +
    geom_point(size = 7, color = "firebrick") +
    geom_text(color = "white", size = 2.5) +
    # Accuracy basal
    labs(title = paste0(title, " - ", "RMSE"),
         caption = "Validation: Mean RMSE repeated-CV",
         x = "") +
    coord_flip() +
    theme_bw()
  
  byMAE <- metrics_resample %>% filter(metric == "MAE") %>%
    group_by(model) %>% 
    summarise(mean = mean(value)) %>%
    ggplot(aes(x = reorder(model, mean), y = mean, label = round(mean, 2))) +
    geom_segment(aes(x = reorder(model, mean), y = 0,
                     xend = model, yend = mean),
                 color = "grey50") +
    geom_point(size = 7, color = "firebrick") +
    geom_text(color = "white", size = 2.5) +
    # Accuracy basal
    labs(title = paste0(title, " - ","MAE"),
         caption = "Validation: Mean RMSE repeated-CV",
         x = "") +
    coord_flip() +
    theme_bw()
  
  byRsquared <- metrics_resample %>% filter(metric == "Rsquared") %>%
    group_by(model) %>% 
    summarise(mean = mean(value)) %>%
    ggplot(aes(x = reorder(model, mean), y = mean, label = round(mean, 2))) +
    geom_segment(aes(x = reorder(model, mean), y = 0,
                     xend = model, yend = mean),
                 color = "grey50") +
    geom_point(size = 7, color = "firebrick") +
    geom_text(color = "white", size = 2.5) +
    # Accuracy basal
    labs(title = paste0(title, " - ","Rsquared"), 
         caption = "Validation: Mean Rsquared repeated-CV",
         x = "")  +
    coord_flip() +
    theme_bw()
  
  
  byRMSE_detail <- metrics_resample %>% filter(metric == "RMSE") %>%
    group_by(model) %>% 
    mutate(mean = mean(value)) %>%
    ungroup() %>%
    ggplot(aes(x = reorder(model, mean), y = value, color = model)) +
    geom_boxplot(alpha = 0.6, outlier.shape = NA) +
    geom_jitter(width = 0.1, alpha = 0.6) +
    theme_bw() +
    labs(title = paste0(title, " - ","RMSE"), 
         caption = "Validation: Mean RMSE repeated-CV",
         x = "")  +
    coord_flip() +
    theme(legend.position = "none")
  
  byMAE_detail <- metrics_resample %>% filter(metric == "MAE") %>%
    group_by(model) %>% 
    mutate(mean = mean(value)) %>%
    ungroup() %>%
    ggplot(aes(x = reorder(model, mean), y = value, color = model)) +
    geom_boxplot(alpha = 0.6, outlier.shape = NA) +
    geom_jitter(width = 0.1, alpha = 0.6) +
    theme_bw() +
    labs(title = paste0(title, " - ","MAE"), 
         caption = "Validation: Mean MAE repeated-CV",
         x = "")  +
    coord_flip() +
    theme(legend.position = "none")
  
  byRsquared_detail <- metrics_resample %>% filter(metric == "MAE") %>%
    group_by(model) %>% 
    mutate(mean = mean(value)) %>%
    ungroup() %>%
    ggplot(aes(x = reorder(model, mean), y = value, color = model)) +
    geom_boxplot(alpha = 0.6, outlier.shape = NA) +
    geom_jitter(width = 0.1, alpha = 0.6) +
    theme_bw() +
    labs(title = paste0(title, " - ","Rsquared"), 
         caption = "Validation: Mean Rsquared repeated-CV",
         x = "")  +
    
    coord_flip() +
    theme(legend.position = "none")
  
  
  return(grid.arrange(byRMSE,  byMAE, byRsquared, byRMSE_detail, byMAE_detail, byRsquared_detail,  ncol = 3))
}

plotResampling.compress <- function(metrics_resample, title = ""){
  
  
  byRMSE <- metrics_resample %>% filter(metric == "RMSE") %>%
    group_by(model) %>% 
    summarise(mean = mean(value)) 
  
  byNRMSE <- metrics_resample %>% filter(metric == "NRMSE") %>%
    group_by(model) %>% 
    summarise(mean = mean(value)) 
  
  byMAE <- metrics_resample %>% filter(metric == "MAE") %>%
    group_by(model) %>% 
    summarise(mean = mean(value)) 
  
  byRsquared <- metrics_resample %>% filter(metric == "Rsquared") %>%
    group_by(model) %>% 
    summarise(mean = mean(value)) 
  
  byRMSE_detail <- metrics_resample %>% filter(metric == "RMSE") %>%
    group_by(model) %>% 
    mutate(mean = mean(value)) %>%
    ungroup() %>%
    ggplot(aes(x = reorder(model, mean), y = value, color = model)) +
    geom_boxplot(alpha = 0.6, outlier.shape = NA) +
    geom_jitter(width = 0.1, alpha = 0.6) +
    
    geom_point(data = byRMSE, aes(x = reorder(model, mean), y = mean, label = round(mean, 2),  color = model), size = 7, shape = 15, alpha = 0.5) +
    geom_text(data = byRMSE, aes(x = reorder(model, mean), y = mean, label = round(mean, 2)),  color = "black", size = 2.5) +
    
    scale_colour_solarized('blue') +
    
    theme_bw() +
    labs(title = paste0(title, " - ","RMSE (Validation)"), 
         subtitle = "Mean RMSE 2x5-XV",
         x = "")  +
    coord_flip() +
    theme(legend.position = "none")
  
  byNRMSE_detail <- metrics_resample %>% filter(metric == "NRMSE") %>%
    group_by(model) %>% 
    mutate(mean = mean(value)) %>%
    ungroup() %>%
    ggplot(aes(x = reorder(model, mean), y = value, color = model)) +
    geom_boxplot(alpha = 0.6, outlier.shape = NA) +
    geom_jitter(width = 0.1, alpha = 0.6) +
    
    geom_point(data = byNRMSE, aes(x = reorder(model, mean), y = mean, label = round(mean, 2),  color = model), size = 7, shape = 15, alpha = 0.5) +
    geom_text(data = byNRMSE, aes(x = reorder(model, mean), y = mean, label = round(mean, 2)),  color = "black", size = 2.5) +
    
    scale_colour_solarized('blue') +
    
    theme_bw() +
    labs(title = paste0(title, " - ","NRMSE (Validation)"), 
         subtitle = "Mean NRMSE 2x5-XV",
         x = "")  +
    coord_flip() +
    theme(legend.position = "none")
  
  byMAE_detail <- metrics_resample %>% filter(metric == "MAE") %>%
    group_by(model) %>% 
    mutate(mean = mean(value)) %>%
    ungroup() %>%
    ggplot(aes(x = reorder(model, mean), y = value, color = model)) +
    geom_boxplot(alpha = 0.6, outlier.shape = NA) +
    geom_jitter(width = 0.1, alpha = 0.6) +
    
    geom_point(data = byMAE, aes(x = reorder(model, mean), y = mean, label = round(mean, 2),  color = model), size = 7, shape = 15, alpha = 0.5) +
    geom_text(data = byMAE, aes(x = reorder(model, mean), y = mean, label = round(mean, 2)),  color = "black", size = 2.5) +
    
    scale_colour_solarized('blue') +
    
    theme_bw() +
    labs(title = paste0(title, " - ","MAE (Validation)"), 
         subtitle = "Mean MAE 2x5-XV",
         x = "")  +
    coord_flip() +
    theme(legend.position = "none")
  
  byRsquared_detail <- metrics_resample %>% filter(metric == "MAE") %>%
    group_by(model) %>% 
    mutate(mean = mean(value)) %>%
    ungroup() %>%
    ggplot(aes(x = reorder(model, mean), y = value, color = model)) +
    geom_boxplot(alpha = 0.6, outlier.shape = NA) +
    geom_jitter(width = 0.1, alpha = 0.6) +
    
    geom_point(data = byRsquared, aes(x = reorder(model, mean), y = mean, label = round(mean, 2),  color = model), size = 7, shape = 15, alpha = 0.5) +
    geom_text(data = byRsquared, aes(x = reorder(model, mean), y = mean, label = round(mean, 2)),  color = "black", size = 2.5) +
    
    scale_colour_solarized('blue') +
    
    theme_bw() +
    labs(title = paste0(title, " - ","Rsquared (Validation)"), 
         subtitle = "Mean Rsquared 2x5-XV",
         x = "")  +
    
    coord_flip() +
    theme(legend.position = "none")
  
  
  return(grid.arrange(byRMSE_detail, byNRMSE_detail, byMAE_detail, ncol = 3))
}

computeSignificances <- function(metrics_resample){
  
  
  matrix_metrics <- metrics_resample %>% filter(metric == "RMSE") %>%
    spread(key = model, value = value) %>%
    select(-Resample, -metric) %>% as.matrix()
  RMSE.friedman <- friedman.test(y = matrix_metrics)$p.value
  
  # print("Friedman.test - MAE")
  matrix_metrics <- metrics_resample %>% filter(metric == "MAE") %>%
    spread(key = model, value = value) %>%
    select(-Resample, -metric) %>% as.matrix()
  MAE.friedman <- friedman.test(y = matrix_metrics)$p.value
  
  # print("Pairwise.wilcox - RMSE")
  metrics_RMSE <- metrics_resample %>% filter(metric == "RMSE")
  comparisonsRMSE  <- pairwise.wilcox.test(x = metrics_RMSE$value, 
                                           g = metrics_RMSE$model,
                                           paired = TRUE,
                                           p.adjust.method = "holm")
  
  comparisonsRMSE <- comparisonsRMSE$p.value %>%
    as.data.frame() %>%
    rownames_to_column(var = "modelA") %>%
    gather(key = "modelB", value = "p_value", -modelA) %>%
    na.omit() %>%
    arrange(modelA) 
  RMSE.wilcox <- comparisonsRMSE
  
  
  # print("Pairwise.wilcox - MAE")
  metrics_MAE <- metrics_resample %>% filter(metric == "MAE")
  comparisonsMAE  <- pairwise.wilcox.test(x = metrics_MAE$value, 
                                          g = metrics_MAE$model,
                                          paired = TRUE,
                                          p.adjust.method = "holm")
  comparisonsMAE <- comparisonsMAE$p.value %>%
    as.data.frame() %>%
    rownames_to_column(var = "modelA") %>%
    gather(key = "modelB", value = "p_value", -modelA) %>%
    na.omit() %>%
    arrange(modelA) 
  MAE.wilcox <- comparisonsMAE
  
  return(list(RMSE.friedman = RMSE.friedman, MAE.friedman = MAE.friedman, RMSE.wilcox = RMSE.wilcox, MAE.wilcox = MAE.wilcox))
  
}

# Test
# -----------------------------------------------------------------------------------------------------

computepredictions <- function(datos_test, models, nameDS = ""){
  
  preds <- extractPrediction(
    models = models,
    testX = datos_test %>% select(-Dffclt),
    testY = datos_test$Dffclt
  )
  return(cbind(nameDS,data.frame(preds)))
}

plotpredictions <- function(preds){
  
  cors <- preds %>% group_by(nameDS, object, dataType) %>% summarise(spearman = cor(obs, pred, method = "spearman"), pearson = cor(obs, pred, method = "pearson"))
  
  cors <- merge(preds, cors)
  
  plot1 <- ggplot(preds, aes(obs, pred, colour = object)) + 
    geom_point() + 
    facet_grid(object ~ dataType, scales = "free") + 
    scale_colour_solarized('blue') + theme_light() + theme(legend.position = "none")
  
  
  plot2 <- plot1 + 
    geom_text(data = cors, aes(x = +Inf, y = -Inf, label = paste0("Spearman: ", round(spearman,3))), hjust = 1.2, vjust = -0.5) +
    geom_text(data = cors, aes(x = +Inf, y = -Inf, label = paste0("Pearson: ", round(pearson,3))), hjust = 1.2, vjust = -2) 
  
  
  return(plot2)
  
}

plotpredictions.metrics <- function(preds, title = ""){
  
  metrics_predsRMSE <- preds %>%
    group_by(object, dataType) %>%
    summarise(RMSE = RMSE(pred,obs)) 
  
  metrics_predsNRMSE <- preds %>%
    group_by(object, dataType) %>%
    summarise(NRMSE = RMSE(pred,obs)/sd(obs)) 
  
  metrics_predsMAE <- preds %>%
    group_by(object, dataType) %>%
    summarise(MAE = MAE(pred,obs)) 
  
  testRMSE <- ggplot(data = metrics_predsRMSE,
                     aes(x = reorder(object, RMSE), y = RMSE,
                         color = dataType, label = round(RMSE, 2))) +
    geom_point(size = 8) +
    scale_color_manual(values = c("#9a031e", "#0f4c5c")) +
    geom_text(color = "white", size = 3) +
    coord_flip() +
    labs(title = paste0(title, " - RMSE"), 
         x = "modelo") +
    theme_bw() + 
    theme(legend.position = "bottom")
  
  testNRMSE <- ggplot(data = metrics_predsNRMSE,
                      aes(x = reorder(object, NRMSE), y = NRMSE,
                          color = dataType, label = round(NRMSE, 2))) +
    geom_point(size = 8) +
    scale_color_manual(values = c("#9a031e", "#0f4c5c")) +
    geom_text(color = "white", size = 3) +
    coord_flip() +
    labs(title = paste0(title, " - NRMSE"), 
         x = "modelo") +
    theme_bw() + 
    theme(legend.position = "bottom")
  
  testMAE <- ggplot(data = metrics_predsMAE,
                    aes(x = reorder(object, MAE), y = MAE,
                        color = dataType, label = round(MAE, 2))) +
    geom_point(size = 8) +
    scale_color_manual(values = c("#9a031e", "#0f4c5c")) +
    geom_text(color = "white", size = 3) +
    coord_flip() +
    labs(title = paste0(title, " - MAE"), 
         x = "modelo") +
    theme_bw() + 
    theme(legend.position = "bottom")
  
  
  return(grid.arrange(testRMSE, testNRMSE, testMAE, ncol = 3))
  
}

plotpredictions.density <- function(preds, title = ""){
  
  return(ggplot(preds) +
           geom_density(aes(obs, fill = "obs"), alpha = 0.5) +
           geom_density(aes(pred, fill = "pred"), alpha = 0.5) + 
           facet_grid(dataType ~ object) + 
           labs(title = title) + xlab("difficulty") + 
           scale_fill_manual(values = c("#0f4c5c", "#9a031e")) +
           theme_light())
  
}


































