library(qshap)

if (requireNamespace("ggplot2", quietly = TRUE)) {
  local({
    loss_matrix <- matrix(
      c(
        -2.0,  0.4,  1.0,
        -0.5, -0.2,  0.3,
         1.5, -1.0, -0.4,
         0.2,  0.1, -0.1
      ),
      nrow = 4L,
      byrow = TRUE,
      dimnames = list(
        c("obs_a", "obs_b", "obs_c", "obs_d"),
        c("first", "second", "third")
      )
    )
    importance <- c(first = 0.2, second = 0.8, third = 0.5)

    device_file <- tempfile(fileext = ".pdf")
    save_file <- tempfile()
    grDevices::pdf(device_file)
    on.exit({
      grDevices::dev.off()
      unlink(c(device_file, paste0(save_file, ".pdf")))
    })

    heatmap_plot <- plot_loss_heatmap(
      loss_matrix,
      global_importance = importance,
      save_name = save_file
    )

    fill_scale <- heatmap_plot$scales$get_scales("fill")
    x_scale <- heatmap_plot$scales$get_scales("x")
    y_scale <- heatmap_plot$scales$get_scales("y")
    raster_data <- heatmap_plot$data

    stopifnot(
      inherits(heatmap_plot, "ggplot"),
      inherits(heatmap_plot$layers[[1L]]$geom, "GeomRaster"),
      inherits(heatmap_plot$layers[[3L]]$geom, "GeomRect"),
      identical(x_scale$labels, c("second", "third", "first", "Total")),
      identical(y_scale$labels, c("obs_a", "obs_b", "obs_c", "obs_d")),
      identical(fill_scale$limits, c(-2, 2)),
      nrow(raster_data) == length(loss_matrix),
      all(c(-2, 1.5) %in% raster_data$contribution),
      raster_data$contribution[
        raster_data$feature_index == 1L &
          raster_data$observation_index == nrow(loss_matrix)
      ] == loss_matrix[4L, 2L],
      nrow(heatmap_plot$layers[[3L]]$data) == nrow(loss_matrix),
      identical(
        heatmap_plot$theme$panel.grid.major.y,
        ggplot2::element_blank()
      ),
      file.exists(paste0(save_file, ".pdf"))
    )

    selection_loss <- cbind(
      a = c(-4, -2, -0.2, 0.1, 2, 5),
      b = rep(0, 6)
    )
    rownames(selection_loss) <- paste0("id_", seq_len(nrow(selection_loss)))
    auto_plot <- plot_loss_heatmap(selection_loss, n_show = 4L)
    id_plot <- plot_loss_heatmap(
      selection_loss,
      samples = c("id_3", "id_1")
    )
    index_plot <- plot_loss_heatmap(selection_loss, samples = c(4L, 2L))
    stopifnot(
      identical(
        auto_plot$scales$get_scales("y")$labels,
        c("id_1", "id_2", "id_5", "id_6")
      ),
      identical(
        id_plot$scales$get_scales("y")$labels,
        c("id_1", "id_3")
      ),
      identical(
        index_plot$scales$get_scales("y")$labels,
        c("id_2", "id_4")
      ),
      nrow(auto_plot$data) == 4L * ncol(selection_loss),
      nrow(id_plot$data) == 2L * ncol(selection_loss)
    )

    local_rsq_matrix <- -loss_matrix / 10
    local_importance <- colSums(local_rsq_matrix)
    local_result <- structure(
      list(
        rsq = local_importance,
        loss = loss_matrix,
        local_rsq = local_rsq_matrix
      ),
      class = c("qshap_rsq", "list")
    )
    dispatched_plot <- plot(local_result, type = "heatmap")
    alias_plot <- plot(local_result, type = "loss_heatmap")
    raw_loss_plot <- plot(local_result, type = "heatmap", quantity = "loss")
    constructed_result <- qshap_result(
      rsq = local_importance,
      feature_names = colnames(loss_matrix),
      n_samples = nrow(loss_matrix),
      loss = loss_matrix,
      local_rsq = local_rsq_matrix
    )
    constructed_plot <- plot_loss_heatmap(constructed_result)
    percent_labels <- dispatched_plot$scales$get_scales("fill")$labels(
      c(-0.01, 0, 0.01)
    )
    expected_percent_labels <- scales::label_percent(accuracy = 0.01)(
      c(-0.01, 0, 0.01)
    )
    stopifnot(
      inherits(dispatched_plot, "ggplot"),
      inherits(alias_plot, "ggplot"),
      inherits(constructed_plot, "ggplot"),
      identical(constructed_result$loss, loss_matrix),
      identical(constructed_result$local_rsq, local_rsq_matrix),
      identical(
        dispatched_plot$scales$get_scales("x")$labels,
        c("first", "second", "third", "Total")
      ),
      identical(
        dispatched_plot$labels$title,
        "Observation-level contributions to the global R² decomposition"
      ),
      identical(
        dispatched_plot$scales$get_scales("fill")$name,
        "Local R² contribution"
      ),
      identical(percent_labels, expected_percent_labels),
      inherits(
        raw_loss_plot$scales$get_scales("fill")$labels,
        "waiver"
      ),
      isTRUE(all.equal(
        sort(dispatched_plot$data$contribution),
        sort(as.vector(local_rsq_matrix)),
        tolerance = 1e-12
      )),
      isTRUE(all.equal(
        sort(raw_loss_plot$data$contribution),
        sort(as.vector(loss_matrix)),
        tolerance = 1e-12
      )),
      identical(
        raw_loss_plot$labels$title,
        "Observation-level loss contributions"
      ),
      isTRUE(all.equal(
        colSums(local_result$local_rsq),
        local_result$rsq,
        tolerance = 1e-12
      ))
    )
  })
}
