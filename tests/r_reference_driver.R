#!/usr/bin/env Rscript
# Drive the susieR reference on the bundled N3finemapping example dataset.
#
# Usage:
#   Rscript r_reference_driver.R <out_dir>
#
# N3finemapping is a simulated fine-mapping dataset with three effect
# variables: individual-level X (574 x 1001), responses Y (574 x 2),
# and the true coefficients true_coef.  We analyse the first response
# column.  Summary statistics for susie_rss are derived from the same X.
#
# Outputs (in out_dir):
#   data_X.tsv            the X matrix (574 x 1001)
#   data_y.tsv            the response vector (first Y column)
#   data_true.tsv         true coefficients (first column)
#   susie_pip.tsv         PIP per variable from susie()
#   susie_alpha.tsv       L x p per-effect alpha matrix from susie()
#   susie_postmean.tsv    posterior mean coefficients from susie()
#   susie_cs.tsv          credible-set membership (variable, cs_label)
#   susie_scalars.tsv     sigma2, elbo, niter, intercept for susie()
#   rss_pip.tsv           PIP per variable from susie_rss()
#   rss_alpha.tsv         L x p per-effect alpha matrix from susie_rss()
#   rss_cs.tsv            credible-set membership for susie_rss()
#   rss_scalars.tsv       sigma2, elbo, niter for susie_rss()
#   rss_z.tsv             the z-scores used by susie_rss()
#   ss_pip.tsv            PIP per variable from susie_suff_stat()

suppressPackageStartupMessages({
  library(susieR)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "R_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data(N3finemapping)
X <- N3finemapping$X
y <- N3finemapping$Y[, 1]
true_coef <- N3finemapping$true_coef[, 1]

# Use a smaller variable window to keep the reference fast and tractable;
# the window is chosen to include all three true effects.
true_idx <- which(true_coef != 0)
keep <- sort(union(true_idx, 1:300))
X <- X[, keep]
true_coef <- true_coef[keep]
p <- ncol(X)
n <- nrow(X)

write.table(X, file.path(out_dir, "data_X.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(data.frame(y = y), file.path(out_dir, "data_y.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(true_coef = true_coef),
            file.path(out_dir, "data_true.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- susie() on individual-level data -------------------------------
fit <- susie(X, y, L = 10)
write.table(data.frame(pip = fit$pip), file.path(out_dir, "susie_pip.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(fit$alpha, file.path(out_dir, "susie_alpha.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(data.frame(postmean = susie_get_posterior_mean(fit)),
            file.path(out_dir, "susie_postmean.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cs_rows <- data.frame(variable = integer(0), cs = character(0))
if (!is.null(fit$sets$cs)) {
  for (i in seq_along(fit$sets$cs)) {
    nm <- names(fit$sets$cs)[i]
    for (v in fit$sets$cs[[i]])
      cs_rows <- rbind(cs_rows, data.frame(variable = v, cs = nm))
  }
}
write.table(cs_rows, file.path(out_dir, "susie_cs.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(sigma2 = fit$sigma2,
                       elbo = fit$elbo[length(fit$elbo)],
                       niter = fit$niter,
                       intercept = fit$intercept),
            file.path(out_dir, "susie_scalars.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- susie_suff_stat() ----------------------------------------------
ss <- compute_suff_stat(X, y, standardize = TRUE)
fit_ss <- susie_suff_stat(XtX = ss$XtX, Xty = ss$Xty, yty = ss$yty,
                          n = ss$n, X_colmeans = ss$X_colmeans,
                          y_mean = ss$y_mean, L = 10)
write.table(data.frame(pip = fit_ss$pip), file.path(out_dir, "ss_pip.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- susie_rss() on summary statistics ------------------------------
ss_uni <- univariate_regression(X, y)
z <- ss_uni$betahat / ss_uni$sebetahat
R <- cor(X)
write.table(data.frame(z = z), file.path(out_dir, "rss_z.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(R, file.path(out_dir, "rss_R.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)

fit_rss <- susie_rss(z = z, R = R, n = n, L = 10)
write.table(data.frame(pip = fit_rss$pip), file.path(out_dir, "rss_pip.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(fit_rss$alpha, file.path(out_dir, "rss_alpha.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
cs_rows <- data.frame(variable = integer(0), cs = character(0))
if (!is.null(fit_rss$sets$cs)) {
  for (i in seq_along(fit_rss$sets$cs)) {
    nm <- names(fit_rss$sets$cs)[i]
    for (v in fit_rss$sets$cs[[i]])
      cs_rows <- rbind(cs_rows, data.frame(variable = v, cs = nm))
  }
}
write.table(cs_rows, file.path(out_dir, "rss_cs.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(sigma2 = fit_rss$sigma2,
                       elbo = fit_rss$elbo[length(fit_rss$elbo)],
                       niter = fit_rss$niter),
            file.path(out_dir, "rss_scalars.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- info -----------------------------------------------------------
write.table(data.frame(n = n, p = p,
                       n_true = sum(true_coef != 0)),
            file.path(out_dir, "info.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("R susieR reference done\n")
