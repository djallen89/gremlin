library(jsonlite)

div <- function(a, b) {
    a / b
}

gigaflops <- function(order, time) {
    return(2 * order^3 / (time * 1.0e9))
}

getdata <- function(version) {
    dirpath = paste("./target/criterion/final_dgemm_", version, sep="")
    all_files <- list.files(path=dirpath,
                            pattern="*sq",
                            full.names=TRUE,
                            recursive=FALSE)

    all_arr <- lapply(all_files, function(dir) {
        dirname <- trimws(dir, "right")

        nfilename <- paste(dirname, "/base/benchmark.json", sep="")
        version_bench_name <- fromJSON(nfilename)$function_id
        version_bench_n <- as.integer(strsplit(version_bench_name, "sq"))
        
        valfilename <- paste(dirname, "/base/estimates.json", sep="")
        version_val <- fromJSON(valfilename)$Mean$point_estimate / 1e9
        flops <- gigaflops(version_bench_n, version_val)
        c(version_bench_n, version_val, flops)
    })

    all <- matrix(unlist(all_arr), ncol = 3, byrow = TRUE)
    colnames(all) <- c("N", "Time (sec)", "GigaFLOPS")
    return(all)
}

arr_mod_n <- function(arr, n) {
    return(arr[arr[, 1] %% n == 0,])
}

arr_nomod512 <- function(arr) {
    rough <- arr[arr[, 1] %% 512 != 0,]
    return(rough)
}

speedup_n <- function(serial, parallel, threads) {
    order <- serial[, 1]
    speedup <- mapply(div, serial[, 2], parallel[, 2])
    return(cbind(order, speedup))
}

efficiency_n <- function(serial, parallel, threads) {
    order <- serial[, 1]
    efficiency <- mapply(div, serial[, 2], parallel[, 2]) / threads * 100;
    return(cbind(order, efficiency))
}

superlinear_n <- function(arr) {
    return(arr[arr[, 2] > 100,])
}

gustafson_n <- function(parallel, n, model) {
    orders_n <- as.integer(round(parallel[,1]))
    times_n <- (model$coefficients[4]*orders_n^3 +
    model$coefficients[3]*orders_n^2 +
    model$coefficients[2]*orders_n +
    model$coefficients[1]) / n
    return(cbind(orders_n, times_n))
}

gust_err <- function(model_all, exp_all) {
    model <- model_all[exp_all[,1] >= 640,]
    end <- length(model_all[,1]) 
    start <- end - length(model[,1]) + 1
    exp <- exp_all[c(start:end),]
    orders_n <- exp[,1]
    err_n <- abs(exp[,2] - model[,2])/model[,2] * 100
    return(cbind(orders_n, err_n))
}

analysis_dir = "./analysis/"
output_dir = paste(analysis_dir, "images/", sep="")
dir.create(analysis_dir)
dir.create(output_dir)

par_plot <- function(name, maintitle, logdim, vertlim, xticks, ylabel,
                     legx, legy, colors, par_pches, 
                     arr2, arr4, arr8, arr16) {
    filename <- paste(output_dir, name, ".pdf", sep="")
    filename
    pdf(filename, height=6, width=8)
    # X11()
    plot(arr2[,1], arr2[,2], log=logdim, ylim=vertlim, bg="lightblue", col=colors[1],
         pch=par_pches[1],
         xlab="N", xaxt='n', ylab=ylabel)
    axis(side=1, at=xticks)
    points(arr4[,1], arr4[,2], bg="lightblue", col=colors[2], pch=par_pches[2])
    points(arr8[,1], arr8[,2], bg="lightblue", col=colors[3], pch=par_pches[3])
    points(arr16[,1], arr16[,2], bg="lightblue", col=colors[4], pch=par_pches[4])
    title(main=maintitle)
    legend(legx, legy, legend=c("2 threads", "4 threads", "8 threads", "16 threads"),
           col=colors,
           pch=par_pch, lty=1:4, cex=1.2)
    dev.off()
}

## All data

ndarray_all <- getdata("ndarray")
serial_all <- getdata("serial")
parallel_all_2 <- getdata("parallel02")
parallel_all_4 <- getdata("parallel04")
parallel_all_8 <- getdata("parallel")
parallel_all_16 <- getdata("parallel16")

## Just the orders which are a multiple of 4 

serial_mod4 <- arr_mod_n(serial_all, 4)
ndarray_mod4 <- arr_mod_n(ndarray_all, 4)
parallel_mod4_2 <- arr_mod_n(parallel_all_2, 4)
parallel_mod4_4 <- arr_mod_n(parallel_all_4, 4)
parallel_mod4_8 <- arr_mod_n(parallel_all_8, 4)
parallel_mod4_16 <- arr_mod_n(parallel_all_16, 4)

## Just the orders which are a multiple of 32 

ndarray_mod32 <- arr_mod_n(ndarray_all, 32)

serial_mod32 <- arr_mod_n(serial_all, 32)
parallel_mod32_2 <- arr_mod_n(parallel_all_2, 32)
parallel_mod32_4 <- arr_mod_n(parallel_all_4, 32)
parallel_mod32_8 <- arr_mod_n(parallel_all_8, 32)
parallel_mod32_16 <- arr_mod_n(parallel_all_16, 32)

# Just the orders which are a multiple of 512 #

serial_mod512 <- arr_mod_n(serial_mod4, 512)
parallel_mod512_2 <- arr_mod_n(parallel_mod4_2, 512)
parallel_mod512_4 <- arr_mod_n(parallel_mod4_4, 512)
parallel_mod512_8 <- arr_mod_n(parallel_mod4_8, 512)
parallel_mod512_16 <- arr_mod_n(parallel_mod4_16, 512)

serial_nomod512 <- arr_nomod512(serial_mod32)
serial_model <- lm(serial_nomod512[,2] ~ poly(serial_nomod512[,1], 3, raw=TRUE))
summary(serial_model)
coef(serial_model)
serial_model_points <- gustafson_n(serial_mod32, 1, serial_model)
fake_model <- lm(serial_model_points[,2] ~ poly(serial_model_points[,1], 3, raw=TRUE))

parallel_nomod512_2 <- arr_nomod512(parallel_mod32_2)
parallel_nomod512_4 <- arr_nomod512(parallel_mod32_4)
parallel_nomod512_8 <- arr_nomod512(parallel_mod32_8)
parallel_nomod512_16 <- arr_nomod512(parallel_mod32_16)

serial_model_512 <- lm(serial_mod512[,2] ~ poly(serial_mod512[,1], 3, raw=TRUE))
summary(serial_model_512)
coef(serial_model_512)

speedup2 <- speedup_n(serial_mod32, parallel_mod32_2)
speedup4 <- speedup_n(serial_mod32, parallel_mod32_4)
speedup8 <- speedup_n(serial_mod32, parallel_mod32_8)
speedup16 <- speedup_n(serial_mod32, parallel_mod32_16)

efficiency2  <- efficiency_n(serial_mod32, parallel_mod32_2, 2)
efficiency4  <- efficiency_n(serial_mod32, parallel_mod32_4, 4)
efficiency8  <- efficiency_n(serial_mod32, parallel_mod32_8, 8)
efficiency16 <- efficiency_n(serial_mod32, parallel_mod32_16, 8)

superlinear2 <- superlinear_n(efficiency2)
superlinear4 <- superlinear_n(efficiency4)
superlinear8 <- superlinear_n(efficiency8)
superlinear16 <- superlinear_n(efficiency16)

gust_err_2 <- gust_err(gustafson_n(serial_nomod512, 2, serial_model), parallel_nomod512_2)
gust_err_4 <- gust_err(gustafson_n(serial_nomod512, 4, serial_model), parallel_nomod512_4)
gust_err_8 <- gust_err(gustafson_n(serial_nomod512, 8, serial_model), parallel_nomod512_8)
gust_err_16 <- gust_err(gustafson_n(serial_nomod512, 8, serial_model), parallel_nomod512_16)

legend_cex <- 1.2

#points(ndarray_mod32[,1], ndarray_mod32[,2], col="purple", pch=4, cex=0.5)
# Comparing 1thread to ndarray
pdf(paste(output_dir, "/mine_v_ndarray.pdf", sep=""), height=5, width=4)
# X11()
plot(ndarray_mod32[,1], ndarray_mod32[,2]/serial_mod32[,2],
     log="", col="black", pch=8,
     xlab="N", ylab="Ratio")
title(main="Ratio of ndarray to gremlin's wall time")
dev.off()

# Time vs n for all threads
#X11()
pdf(paste(output_dir, "time_v_n.pdf", sep=""), height=6, width=8)
plot(serial_mod32[,1], serial_mod32[,2], log="xy", col="black", pch=20, bg="lightblue",
     xlab="N", xaxt='n', ylim=c(16e-6, 20),
     ylab="Time (sec)", yaxt='n', cex=0.55)
axis(side=2, at=c(16e-6, 20e-4, 20e-2, 20))
axis(side=1, at=c(64, 128, 256, 512, 1024, 2048, 5248))
points(parallel_mod32_2[,1], parallel_mod32_2[,2], col="darkgreen", pch=24, bg="lightblue", cex=0.55)
points(parallel_mod32_4[,1], parallel_mod32_4[,2], col="blue", pch=22, bg="lightblue", cex=0.55)
points(parallel_mod32_8[,1], parallel_mod32_8[,2], col="red", pch=23, bg="lightblue", cex=0.55)
points(parallel_mod32_16[,1], parallel_mod32_16[,2], col="black", pch=25, bg="lightblue",cex=0.55)
title(main="Wall Time to compute C = AB + C")
legend(85, 8, legend=c("1 thread", "2 threads", "4 threads", "8 threads", "16 threads"),
       col=c("black", "darkgreen", "blue", "red", "black"),
       pch=c(20, 24, 22, 23, 25), lty=1:4, cex=legend_cex)

dev.off()

pdf(paste(output_dir, "gigaflops_v_n.pdf", sep=""), height=6, width=8)
plot(ndarray_mod32[,1], ndarray_mod32[,3], log="x", col="purple", pch=8, bg="lightblue",
     xaxt="n", xlab="N", ylab="GigaFLOPS per core", ylim=c(0, 25), cex=0.55)
axis(side=1, at=c(64, 128, 256, 512, 1024, 2048, 5248))
points(serial_mod32[,1], serial_mod32[,3], col="black", pch=20, bg="lightblue", cex=0.55)
points(parallel_mod32_4[,1], parallel_mod32_4[,3] / 4, col="blue", pch=22, bg="lightblue", cex=0.55)
points(parallel_mod32_8[,1], parallel_mod32_8[,3] / 8, col="red", pch=23, bg="lightblue", cex=0.55)
title(main="Computational Intensity")
legend(1500, 10, legend=c("ndarray", "1 thread", "4 threads", "8 threads"),
       col=c("purple", "black", "blue", "red"),
       pch=c(8, 20, 22, 23), lty=1:4, cex=legend_cex)
dev.off()

par_legend <- c("2 threads", "4 threads", "8 threads", "16 threads")
par_colors <- c("darkgreen", "blue", "red", "black")
par_pch <- c(24, 22, 23, 25)

# Speedup
par_plot("speedup_v_n", "Speedup vs Order", "x", c(0, 8), c(64, 256, 1024, 2048, 5248),
         "Speedup", 64, 8, par_colors, par_pch,
         speedup2, speedup4, speedup8, speedup16)

# Efficiency
# pdf("../doc/images/eff_v_n.pdf", height=6, width=8)
par_plot("eff_v_n", "Efficiency vs Order", "x", c(15, 101), c(64, 512, 1536, 5248),
         "Efficiency (%)", 500, 65, par_colors, par_pch,
         efficiency2, efficiency4, efficiency8, efficiency16)

par_plot("gust_err", "Gustafson Model Prediction Error", "x", c(0, 30),
         c(640, 1024, 2048, 5248), "Error (%)", 1500, 20, par_colors, par_pch,
         gust_err_2, gust_err_4, gust_err_8, gust_err_16)
