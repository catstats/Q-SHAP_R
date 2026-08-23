library(qshap)

if (requireNamespace("ggplot2", quietly = TRUE)) {
  local({
    device_file <- tempfile(fileext = ".pdf")
    elbow_file <- tempfile()
    cumulative_file <- tempfile()
    grDevices::pdf(device_file)
    on.exit({
      grDevices::dev.off()
      unlink(c(
        device_file,
        paste0(elbow_file, ".pdf"),
        paste0(cumulative_file, ".pdf")
      ))
    })

    result <- structure(
      list(rsq = c(0.4, 0.2, 0.1)),
      class = "qshap_rsq"
    )

    rsq_plot <- plot(result)
    plot(result, type = "elbow", save_name = elbow_file)
    elbow_plot <- ggplot2::last_plot()
    cumulative_plot <- plot(result, type = "cumu", save_name = cumulative_file)

    stopifnot(
      inherits(rsq_plot, "ggplot"),
      identical(rsq_plot$theme$text$family, "sans"),
      identical(rsq_plot$theme$plot.title$face, "bold"),
      identical(rsq_plot$theme$axis.title$face, "bold"),
      identical(rsq_plot$theme$axis.text$face, "plain"),
      grepl("bold", paste(deparse(rsq_plot$labels$title), collapse = "")),
      identical(elbow_plot$labels$title, "Explained variance by top features"),
      identical(elbow_plot$labels$y, "Explained variance"),
      identical(cumulative_plot$labels$title, "Cumulative explained variance by top features"),
      identical(cumulative_plot$labels$y, "Cumulative explained variance"),
      identical(cumulative_plot$layers[[2]]$geom_params$check_overlap, FALSE),
      "label_y" %in% names(cumulative_plot$layers[[2]]$data),
      nrow(cumulative_plot$layers[[2]]$data) == 3L,
      all(grepl("\n", cumulative_plot$layers[[2]]$data$lab, fixed = TRUE)),
      !"inc_threshold" %in% names(formals(qshap:::vis$cumu))
    )

    stopifnot(file.exists(paste0(elbow_file, ".pdf")))
    stopifnot(file.exists(paste0(cumulative_file, ".pdf")))
  })
}
