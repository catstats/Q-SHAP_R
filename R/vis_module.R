#' @import ggplot2
#' @importFrom scales label_percent
NULL

#' Visualization Module for Q-SHAP Results
#' 
#' An environment containing visualization functions for Q-SHAP results.
#' Access functions using \code{vis$rsq()}, \code{vis$elbow()}, etc.
#' 
#' @format An environment with visualization functions:
#' \describe{
#'   \item{rsq}{Bar plot of feature-specific R-squared values}
#'   \item{elbow}{Elbow plot showing top contributing features}
#'   \item{cumu}{Cumulative explained variance plot}
#'   \item{gcorr}{Generalized correlation plot (square root of R-squared)}
#'   \item{hist}{Histogram of feature-specific R-squared contributions}
#'   \item{density}{Density plot of feature-specific R-squared contributions}
#'   \item{loss}{Interactive loss explorer (requires shiny)}
#'   \item{loss_heatmap}{Heatmap of observation-level contributions}
#' }
#' 
#' @keywords internal
vis <- new.env(parent = emptyenv())

# helper: pick a "matplotlib-like" palette name
.vis_palette <- function(name = "viridis", n = 256) {
  nm <- tolower(name)
  if (nm %in% c("viridis","magma","inferno","plasma","cividis","turbo")) {
    return(viridisLite::viridis(n, option = nm))
  }
  if (nm %in% c("blues","greens","reds","purples","oranges")) {
    return(grDevices::hcl.colors(n, palette = tools::toTitleCase(nm)))
  }
  if (nm %in% c("pastel1","pastel2")) {
    return(grDevices::hcl.colors(n, palette = paste("Pastel", substr(nm, 7, 7))))
  }
  viridisLite::viridis(n)
}

#' Plot method for qshap_rsq objects
#'
#' This S3 method enables `plot(x, ...)` where `x` is a `qshap_rsq` object.
#' It dispatches to the visualization functions in `vis`.
#'
#' @param x A `qshap_rsq` object.
#' @param y Not used.
#' @param type Plot type: one of "rsq", "elbow", "cumu", "gcorr", "hist",
#'   "density", "loss", or "heatmap". `"loss_heatmap"` is retained as an
#'   alias for `"heatmap"`.
#' @param ... Passed to the underlying visualization function.
#'
#' @return A ggplot2 object (invisibly).
#'
#' @method plot qshap_rsq
#' @export
plot.qshap_rsq <- function(x, y = NULL,
                          type = c("rsq", "elbow", "cumu", "gcorr", "hist", "density", "loss", "heatmap", "loss_heatmap"), ...) {
  type <- match.arg(type)

  # Try common field names first
  rsq_values <- NULL
  if (!is.null(x$rsq)) {
    rsq_values <- x$rsq
  } else if (!is.null(x$phi_rsq)) {
    rsq_values <- x$phi_rsq
  } else if (!is.null(x$contrib)) {
    rsq_values <- x$contrib
  } else if (is.numeric(x) && is.vector(x)) {
    rsq_values <- x
  } else {
    # fallback: first numeric vector element in the list
    num_elts <- vapply(x, function(z) is.numeric(z) && is.vector(z), logical(1))
    if (any(num_elts)) rsq_values <- x[[which(num_elts)[1]]]
  }

  if (identical(type, "loss")) {
    if (!is.null(x$loss)) {
      return(invisible(vis$loss(x$loss, ...)))
    }
    stop("type='loss' requires a qshap_rsq object with a $loss matrix (run qshap_rsq(..., local=TRUE)).")
  }

  if (type %in% c("heatmap", "loss_heatmap")) {
    if (is.null(x$local_rsq) && is.null(x$loss)) {
      stop(
        "type='heatmap' requires a qshap_rsq object with $local_rsq or $loss ",
        "(run qshap_rsq(..., local=TRUE))."
      )
    }
    return(invisible(plot_loss_heatmap(x, ...)))
  }

  if (is.null(rsq_values)) {
    stop("Cannot find R^2 contributions in this qshap_rsq object. Expected one of: $rsq, $phi_rsq, $contrib.")
  }

  rsq_values <- as.numeric(rsq_values)

  invisible(
    switch(type,
           rsq = vis$rsq(rsq_values, ...),
           elbow = vis$elbow(rsq_values, ...),
           cumu = vis$cumu(rsq_values, ...),
           gcorr = vis$gcorr(rsq_values, ...),
           hist = vis$hist(rsq_values, ...),
           density = vis$density(rsq_values, ...))
  )
}

#' @export
print.qshap_rsq <- function(x, ...) {
  cat("qshap_rsq object\n")
  if (!is.null(x$rsq)) {
    cat("- rsq length:", length(x$rsq), "\n")
  }
  if (!is.null(x$loss)) {
    cat("- loss dim:", paste(dim(x$loss), collapse = " x "), "\n")
  }
  if (!is.null(x$local_rsq)) {
    cat("- local_rsq dim:", paste(dim(x$local_rsq), collapse = " x "), "\n")
  }
  invisible(x)
}

#' Summary method for qshap_rsq objects
#'
#' Provides a summary of the qshap_rsq object, showing the top features by R-squared contribution
#'
#' @param object A \code{qshap_rsq} object
#' @param n Integer number of top features to display (default: 10)
#' @param ... Additional arguments (currently unused)
#' @return The input \code{object} is returned invisibly. Called primarily for
#'   its side effect of printing a summary of the \code{qshap_rsq} object to
#'   the console.
#' @export
summary.qshap_rsq <- function(object, n = 10, ...) {
  cat("Q-SHAP R^2 Summary\n")
  cat("=================\n\n")
  
  if (is.null(object$rsq)) {
    cat("No R^2 values available in this object.\n")
    return(invisible(object))
  }
  
  rsq <- object$rsq
  total_rsq <- sum(rsq, na.rm = TRUE)
  
  cat("Overall Statistics:\n")
  cat("  Total R^2:", round(total_rsq, 6), "\n")
  cat("  Number of features:", length(rsq), "\n")
  
  # Add SD information if available
  if (!is.null(object$sd_rsq)) {
    cat("  Standard errors: Available\n")
  }
  if (!is.null(object$loss)) {
    cat("  Loss matrix dim:", paste(dim(object$loss), collapse = " x "), "\n")
  }
  if (!is.null(object$local_rsq)) {
    cat(
      "  Local R^2-scale contribution matrix dim:",
      paste(dim(object$local_rsq), collapse = " x "),
      "\n"
    )
  }
  
  cat("\nR^2 Distribution:\n")
  cat("  Min:", round(min(rsq, na.rm = TRUE), 6), "\n")
  cat("  Q1:", round(stats::quantile(rsq, 0.25, na.rm = TRUE), 6), "\n")
  cat("  Median:", round(stats::median(rsq, na.rm = TRUE), 6), "\n")
  cat("  Mean:", round(mean(rsq, na.rm = TRUE), 6), "\n")
  cat("  Q3:", round(stats::quantile(rsq, 0.75, na.rm = TRUE), 6), "\n")
  cat("  Max:", round(max(rsq, na.rm = TRUE), 6), "\n")
  
  # Count significant features
  sig_features <- sum(rsq > 0.01, na.rm = TRUE)
  cat("\nSignificant Features (R^2 > 0.01):", sig_features, "\n")
  
  # Show top N features
  cat("\nTop", min(n, length(rsq)), "features by R^2:\n")
  
  # Get feature names if available (from names attribute)
  feature_names <- names(rsq)
  if (is.null(feature_names)) {
    feature_names <- paste0("Feature_", seq_along(rsq))
  }
  
  # Create data frame and sort
  df <- data.frame(
    Feature = feature_names,
    R_squared = rsq,
    stringsAsFactors = FALSE
  )
  
  # Add SD column if available
  if (!is.null(object$sd_rsq)) {
    df$SE <- object$sd_rsq
  }
  
  df <- df[order(df$R_squared, decreasing = TRUE), ]
  df <- utils::head(df, n)
  
  # Print as formatted table
  print(df, row.names = FALSE, digits = 4)
  
  if (length(rsq) > n) {
    cat("... and", length(rsq) - n, "more features\n")
  }
  
  invisible(object)
}

# helper: ensure palette maps low->light and high->dark (so large values are darker)
.vis_high_dark <- function(pal) {
  if (length(pal) < 2) return(pal)
  lum <- function(col) {
    rgb <- grDevices::col2rgb(col) / 255
    # relative luminance (WCAG)
    0.2126 * rgb[1, ] + 0.7152 * rgb[2, ] + 0.0722 * rgb[3, ]
  }
  l1 <- lum(pal[1])
  lN <- lum(pal[length(pal)])
  # if palette goes dark->light (end is lighter), reverse it
  if (lN > l1) rev(pal) else pal
}

.vis_font_family <- "sans"

.vis_theme <- function(rotation = 0, horizontal = FALSE, plot_margin = NULL) {
  out <- theme_classic(base_size = 12, base_family = .vis_font_family) +
    theme(
      text = element_text(
        family = .vis_font_family,
        face = "plain",
        color = "black"
      ),
      plot.title = element_text(
        family = .vis_font_family,
        face = "bold",
        size = rel(1.2),
        hjust = 0
      ),
      axis.title = element_text(
        family = .vis_font_family,
        face = "bold",
        size = rel(1)
      ),
      axis.text = element_text(
        family = .vis_font_family,
        face = "plain",
        size = rel(0.8),
        color = "grey30"
      ),
      axis.text.x = element_text(
        angle = if (!horizontal) rotation else 0,
        hjust = 1,
        vjust = 1
      ),
      axis.text.y = element_text(
        angle = if (horizontal) rotation else 0,
        hjust = 1,
        vjust = 0.5
      ),
      legend.title = element_text(
        family = .vis_font_family,
        face = "bold",
        size = rel(1)
      ),
      legend.text = element_text(
        family = .vis_font_family,
        face = "plain",
        size = rel(0.8)
      ),
      panel.grid.major.y = element_line(color = "grey85"),
      panel.grid.minor = element_blank()
    )

  if (!is.null(plot_margin)) {
    out <- out + theme(plot.margin = plot_margin)
  }

  out
}

# Bar plot for Shapley R^2 (or any 1d contribution vector)
vis$rsq <- function(
  x,
  color_map_name = "Blues",
  horizontal = FALSE,
  model_rsq = TRUE,
  max_feature = 10,
  cutoff = 0,
  title = expression(bold(paste("Shapley ", R^2))),
  xtitle = "Feature",
  ytitle = expression(bold(R^2)),
  rotation = 0,
  label = NULL,
  decimal = 3,
  show_value = TRUE,
  save_name = NULL
) {
  x <- as.numeric(x)
  x_sum <- sum(x, na.rm = TRUE)
  x_len <- length(x)

  # how many to show
  cutoff_feature <- sum(x >= cutoff, na.rm = TRUE)
  show_len <- min(x_len, max_feature, cutoff_feature)
  if (show_len <= 0) stop("No features pass the cutoff.")

  # sort desc, keep indices
  ord <- order(x, decreasing = TRUE)
  ord <- ord[seq_len(show_len)]
  sorted_x <- x[ord]

  if (!is.null(label)) {
    if (length(label) != length(x)) stop("label length must match x length.")
    sorted_label <- as.character(label[ord])
  } else {
    sorted_label <- as.character(ord)  # default: 1-based feature index (R style)
  }

  df <- data.frame(
    feature = factor(sorted_label, levels = sorted_label),
    value = sorted_x
  )
 if (horizontal) {
   df$feature <- factor(df$feature, levels = rev(levels(df$feature)))
  }
  # color based on value normalized
  rng <- range(df$value, na.rm = TRUE)
  if (diff(rng) < 1e-12) {
    df$val_norm <- 0.5
  } else {
    df$val_norm <- (df$value - rng[1]) / diff(rng)
  }

  # ensure larger values map to darker colors, independent of palette direction
  pal <- .vis_high_dark(.vis_palette(color_map_name, n = 256))
  df$fill <- pal[pmax(1, pmin(256, 1 + floor(df$val_norm * 255)))]

  # label text (avoid showing '-0.000' from floating-point noise)
  val_for_txt <- df$value
  tol <- 0.5 * 10^(-decimal)
  val_for_txt[abs(val_for_txt) < tol] <- 0
  df$txt <- formatC(val_for_txt, format = "f", digits = decimal)

  # add numeric labels on bars
  if (isTRUE(show_value)) {
    if (!horizontal) {
      # label above the bar
      p_label_layer <- geom_text(
        aes(label = txt),
        vjust = -0.35,
        size = 3.6,
        family = .vis_font_family,
        fontface = "plain"
      )
    } else {
      # after coord_flip(), x/y swap; use hjust to place label to the right of the bar
      p_label_layer <- geom_text(
        aes(label = txt),
        hjust = -0.15,
        size = 3.6,
        family = .vis_font_family,
        fontface = "plain"
      )
    }
  }

  # publication-ready plot
  p <- ggplot(df, aes(x = feature, y = value)) +
    geom_col(aes(fill = fill), width = 0.8, show.legend = FALSE) +
    scale_fill_identity() +
    { if (isTRUE(show_value)) p_label_layer else NULL } +
    scale_y_continuous(expand = ggplot2::expansion(mult = c(0.02, 0.14))) +
    labs(title = title, x = xtitle, y = ytitle) +
    .vis_theme(rotation = rotation, horizontal = horizontal)

  if (horizontal) {
    p <- p + coord_flip(clip = "off")
  } else {
    # allow labels to extend slightly above the panel
    p <- p + coord_cartesian(clip = "off")
  }

  # add model rsq annotation
  if (model_rsq) {
    ann <- deparse(bquote("Model " * R^2 * ": " * .(formatC(x_sum, format = "f", digits = 3))))
    p <- p + annotate("text",
      x = if (!horizontal) show_len else 1,
      y = max(df$value, na.rm = TRUE),
      label = ann,
      hjust = 1, vjust = 1,
      size = 4,
      family = .vis_font_family,
      parse = TRUE
    )
  }

  if (!is.null(save_name)) {
    ggsave(filename = paste0(save_name, ".pdf"), plot = p, width = 7, height = 4.2)
  }

  print(p)
  invisible(p)
}

# Interactive loss explorer (like ipywidgets) using shiny
# loss: n x p matrix
vis$loss <- function(
  loss,
  save_ind = NULL,
  save_prefix = "Shapley loss sample",
  title = "Shapley loss: sample",
  color_map_name = "Blues",
  model_rsq = FALSE,
  decimal = 0,
  xtitle = "Feature index (1-based)",
  ytitle = "Loss"
) {
  loss <- as.matrix(loss)

  if (!is.null(save_ind)) {
    save_name <- paste0(save_prefix, " ", save_ind)
    vis$rsq(
      loss[save_ind, ],
      title = paste0(title, " ", save_ind),
      color_map_name = color_map_name,
      model_rsq = model_rsq,
      decimal = decimal,
      xtitle = xtitle,
      ytitle = ytitle,
      save_name = save_name
    )
    return(invisible(NULL))
  }

  # interactive shiny mini-app
  ui <- shiny::fluidPage(
    shiny::titlePanel(title),
    shiny::sidebarLayout(
      shiny::sidebarPanel(
        shiny::numericInput("i", "Sample index", value = 1, min = 1, max = nrow(loss), step = 1),
        shiny::checkboxInput("horizontal", "Horizontal", value = FALSE),
        shiny::numericInput("max_feature", "Maximum features", value = min(10, ncol(loss)), min = 1, max = ncol(loss), step = 1),
        shiny::numericInput("cutoff", "Cutoff", value = 0, step = 0.01)
      ),
      shiny::mainPanel(
        shiny::plotOutput("plt", height = "420px")
      )
    )
  )

  server <- function(input, output, session) {
    output$plt <- shiny::renderPlot({
      i <- as.integer(input$i)
      i <- max(1L, min(nrow(loss), i))
      vis$rsq(
        loss[i, ],
        title = paste0(title, " ", i),
        color_map_name = color_map_name,
        model_rsq = model_rsq,
        decimal = decimal,
        xtitle = xtitle,
        ytitle = ytitle,
        horizontal = isTRUE(input$horizontal),
        max_feature = as.integer(input$max_feature),
        cutoff = as.numeric(input$cutoff)
      )
    })
  }

  shiny::shinyApp(ui, server)
}

# Observation-level loss heatmap
vis$loss_heatmap <- function(
  loss,
  global_importance = NULL,
  feature_names = NULL,
  samples = NULL,
  n_show = 40,
  title = "Observation-level loss contributions",
  xtitle = "Feature",
  ytitle = "Observations",
  legend_title = "Local contribution",
  low_color = "#2166AC",
  mid_color = "white",
  high_color = "#B2182B",
  rotation = 45,
  percent_scale = FALSE,
  save_name = NULL
) {
  if (length(dim(loss)) != 2L) {
    stop("loss must be a two-dimensional matrix or data frame.")
  }
  loss <- as.matrix(loss)
  if (!is.numeric(loss)) stop("loss must contain numeric values.")
  if (nrow(loss) < 1L || ncol(loss) < 1L) {
    stop("loss must contain at least one observation and one feature.")
  }

  n_obs <- nrow(loss)
  n_feature <- ncol(loss)
  column_names <- colnames(loss)
  observation_names <- rownames(loss)
  observation_ids <- if (is.null(observation_names)) {
    as.character(seq_len(n_obs))
  } else {
    as.character(observation_names)
  }

  if (is.null(feature_names)) {
    feature_names <- column_names
    if (is.null(feature_names)) {
      feature_names <- paste0("Feature ", seq_len(n_feature))
    }
  }
  if (length(feature_names) != n_feature) {
    stop("feature_names length must match the number of loss columns.")
  }
  feature_names <- as.character(feature_names)

  if (is.null(global_importance)) {
    # Q-SHAP global importance is proportional to the negative column sum of
    # the observation-level loss decomposition.
    global_importance <- -colSums(loss, na.rm = TRUE)
  } else {
    importance_names <- names(global_importance)
    if (length(global_importance) != n_feature) {
      stop("global_importance length must match the number of loss columns.")
    }
    if (!is.null(importance_names) && !is.null(column_names) &&
        all(column_names %in% importance_names)) {
      global_importance <- global_importance[match(column_names, importance_names)]
    }
  }
  global_importance <- as.numeric(global_importance)
  feature_order <- order(
    global_importance,
    decreasing = TRUE,
    na.last = TRUE,
    method = "radix"
  )

  observation_total <- rowSums(loss, na.rm = TRUE)
  if (is.null(samples)) {
    if (length(n_show) != 1L || !is.numeric(n_show) || !is.finite(n_show) ||
        n_show < 1 || n_show != as.integer(n_show)) {
      stop("n_show must be a positive integer.")
    }
    n_select <- min(as.integer(n_show), n_obs)
    n_high <- ceiling(n_select / 2)
    n_low <- floor(n_select / 2)

    high_order <- order(
      observation_total,
      decreasing = TRUE,
      na.last = TRUE,
      method = "radix"
    )
    low_order <- order(
      observation_total,
      decreasing = FALSE,
      na.last = TRUE,
      method = "radix"
    )
    selected_observations <- utils::head(high_order, n_high)
    low_order <- low_order[!low_order %in% selected_observations]
    selected_observations <- c(
      selected_observations,
      utils::head(low_order, n_low)
    )

    if (length(selected_observations) < n_select) {
      remaining <- setdiff(
        order(abs(observation_total), decreasing = TRUE, na.last = TRUE),
        selected_observations
      )
      selected_observations <- c(
        selected_observations,
        utils::head(remaining, n_select - length(selected_observations))
      )
    }
    selected_observations <- selected_observations[
      order(
        observation_total[selected_observations],
        decreasing = TRUE,
        na.last = TRUE,
        method = "radix"
      )
    ]
  } else {
    if (is.factor(samples)) samples <- as.character(samples)
    if (is.numeric(samples)) {
      if (length(samples) < 1L || any(!is.finite(samples)) ||
          any(samples != as.integer(samples)) ||
          any(samples < 1L | samples > n_obs)) {
        stop("Numeric samples must contain valid observation indices.")
      }
      selected_observations <- as.integer(samples)
    } else if (is.character(samples)) {
      if (is.null(observation_names)) {
        stop("Character samples require row names/IDs on the loss matrix.")
      }
      if (anyDuplicated(observation_names)) {
        stop("Character samples require unique row names/IDs.")
      }
      selected_observations <- match(samples, observation_names)
      if (anyNA(selected_observations)) {
        unknown <- unique(samples[is.na(selected_observations)])
        stop("Unknown observation ID(s): ", paste(unknown, collapse = ", "))
      }
    } else {
      stop("samples must be NULL, numeric observation indices, or character row names/IDs.")
    }
    selected_observations <- selected_observations[!duplicated(selected_observations)]
  }

  selected_loss <- loss[selected_observations, feature_order, drop = FALSE]
  selected_total <- observation_total[selected_observations]
  selected_ids <- observation_ids[selected_observations]
  n_selected <- length(selected_observations)
  ordered_feature_names <- feature_names[feature_order]
  plot_data <- data.frame(
    feature_index = rep(seq_len(n_feature), each = n_selected),
    observation_index = rep(rev(seq_len(n_selected)), times = n_feature),
    contribution = as.vector(selected_loss)
  )
  total_x <- n_feature + 1.25
  total_data <- data.frame(
    observation_index = rev(seq_len(n_selected)),
    contribution = selected_total,
    xmin = total_x - 0.18,
    xmax = total_x + 0.18,
    ymin = rev(seq_len(n_selected)) - 0.5,
    ymax = rev(seq_len(n_selected)) + 0.5
  )

  finite_loss <- c(
    selected_loss[is.finite(selected_loss)],
    selected_total[is.finite(selected_total)]
  )
  color_limit <- if (length(finite_loss) > 0L) max(abs(finite_loss)) else 0
  if (!is.finite(color_limit) || color_limit == 0) color_limit <- 1

  p <- ggplot(
    plot_data,
    aes(x = feature_index, y = observation_index, fill = contribution)
  ) +
    geom_raster(interpolate = FALSE) +
    geom_vline(
      xintercept = n_feature + 0.78,
      linewidth = 0.4,
      color = "grey60"
    ) +
    geom_rect(
      data = total_data,
      aes(
        xmin = xmin,
        xmax = xmax,
        ymin = ymin,
        ymax = ymax,
        fill = contribution
      ),
      inherit.aes = FALSE
    ) +
    scale_fill_gradient2(
      name = legend_title,
      low = low_color,
      mid = mid_color,
      high = high_color,
      midpoint = 0,
      limits = c(-color_limit, color_limit),
      labels = if (isTRUE(percent_scale)) {
        scales::label_percent(accuracy = 0.01)
      } else {
        ggplot2::waiver()
      },
      na.value = "grey90"
    ) +
    scale_x_continuous(
      breaks = c(seq_len(n_feature), total_x),
      labels = c(ordered_feature_names, "Total"),
      limits = c(0.5, total_x + 0.45),
      expand = ggplot2::expansion(mult = 0)
    ) +
    scale_y_continuous(
      breaks = seq_len(n_selected),
      labels = rev(selected_ids),
      limits = c(0.5, n_selected + 0.5),
      expand = ggplot2::expansion(mult = 0)
    ) +
    labs(title = title, x = xtitle, y = ytitle) +
    .vis_theme(
      rotation = rotation,
      plot_margin = ggplot2::margin(t = 10, r = 18, b = 18, l = 45)
    ) +
    theme(
      axis.ticks.y = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank()
    )

  if (!is.null(save_name)) {
    plot_height <- max(5, min(12, 2.5 + 0.16 * n_selected))
    ggsave(
      filename = paste0(save_name, ".pdf"),
      plot = p,
      width = 8.5,
      height = plot_height
    )
  }

  print(p)
  invisible(p)
}

#' Plot observation-level contributions
#'
#' Draws an efficient raster heatmap for selected observations, with a narrow
#' total-contribution strip. For a local Q-SHAP result, the default display is
#' `local_rsq`: observation-level contributions to the global R-squared
#' decomposition, with legend values formatted as percentages. The underlying
#' values remain unchanged. The diverging fill scale is centered at zero.
#'
#' @param loss A numeric contribution matrix with observations in rows and
#'   features in columns, or a `qshap_rsq`/`qshap_result` object produced with
#'   `local = TRUE`.
#' @param global_importance Optional numeric vector used to order features from
#'   most to least globally important. For a `qshap_rsq` object, `$rsq` is used
#'   automatically. For a matrix, the default is `-colSums(loss)` when
#'   `quantity = "loss"` and `colSums(loss)` when `quantity = "local_rsq"`.
#' @param feature_names Optional feature labels. Defaults to `colnames(loss)`.
#' @param quantity Quantity to display when `loss` is a Q-SHAP result:
#'   `"local_rsq"` for local contributions on the R-squared scale or
#'   `"loss"` for raw contributions to the change in squared loss. The
#'   default uses `local_rsq` when available and otherwise falls back to
#'   `loss`. For a matrix, this argument controls labels and the default
#'   feature-ordering rule; the matrix itself is displayed unchanged.
#' @param samples Optional observation indices or row names/IDs to display. If
#'   `NULL`, informative observations are selected automatically from both
#'   extremes of the row sums.
#' @param n_show Total number of observations selected automatically when
#'   `samples = NULL`. Approximately half are taken from each extreme.
#' @param title Optional plot title. For `local_rsq`, the default is
#'   "Observation-level contributions to the global R^2 decomposition".
#' @param xtitle X-axis title.
#' @param ytitle Y-axis title.
#' @param legend_title Optional fill legend title. Defaults to
#'   "Local R^2 contribution" for `local_rsq`.
#' @param low_color Color for negative contributions.
#' @param mid_color Color for zero.
#' @param high_color Color for positive contributions.
#' @param rotation Rotation angle for feature labels.
#' @param save_name Optional output path without the `.pdf` extension.
#'
#' @return A ggplot2 object, invisibly.
#' @export
plot_loss_heatmap <- function(
  loss,
  global_importance = NULL,
  feature_names = NULL,
  quantity = NULL,
  samples = NULL,
  n_show = 40,
  title = NULL,
  xtitle = "Feature",
  ytitle = "Observations",
  legend_title = NULL,
  low_color = "#2166AC",
  mid_color = "white",
  high_color = "#B2182B",
  rotation = 45,
  save_name = NULL
) {
  if (!is.null(quantity)) {
    quantity <- match.arg(quantity, c("local_rsq", "loss"))
  }

  if (inherits(loss, "qshap_rsq") || inherits(loss, "qshap_result")) {
    result <- loss
    if (is.null(quantity)) {
      quantity <- if (!is.null(result$local_rsq)) "local_rsq" else "loss"
    }
    if (is.null(result[[quantity]])) {
      stop(
        "The Q-SHAP result has no $", quantity,
        " matrix; rerun rsq(..., local=TRUE)."
      )
    }
    loss <- result[[quantity]]
    if (is.null(global_importance) && !is.null(result$rsq)) {
      global_importance <- result$rsq
    }
    if (is.null(feature_names) && !is.null(result$feature_names)) {
      feature_names <- result$feature_names
    }
  } else if (is.null(quantity)) {
    quantity <- "loss"
  }

  if (is.null(global_importance) && identical(quantity, "local_rsq")) {
    global_importance <- colSums(as.matrix(loss), na.rm = TRUE)
  }

  if (is.null(title)) {
    title <- if (identical(quantity, "local_rsq")) {
      "Observation-level contributions to the global R\u00b2 decomposition"
    } else {
      "Observation-level loss contributions"
    }
  }
  if (is.null(legend_title)) {
    legend_title <- if (identical(quantity, "local_rsq")) {
      "Local R\u00b2 contribution"
    } else {
      "Local contribution"
    }
  }

  vis$loss_heatmap(
    loss = loss,
    global_importance = global_importance,
    feature_names = feature_names,
    samples = samples,
    n_show = n_show,
    title = title,
    xtitle = xtitle,
    ytitle = ytitle,
    legend_title = legend_title,
    low_color = low_color,
    mid_color = mid_color,
    high_color = high_color,
    rotation = rotation,
    percent_scale = identical(quantity, "local_rsq"),
    save_name = save_name
  )
}

# Elbow plot: top contributions (sorted)
vis$elbow <- function(
  x,
  xtitle = "Top-k features",
  ytitle = "Explained variance",
  max_comp = 10,
  title = "Explained variance by top features",
  label = NULL,
  rotation = 0,
  point_color = "black",
  save_name = NULL
) {
  x <- as.numeric(x)
  max_comp <- min(as.integer(max_comp), length(x))

  ord <- order(x, decreasing = TRUE)
  sel <- ord[seq_len(max_comp)]
  vals <- x[sel]

  # optional labels for selected features (must match length of x)
  if (!is.null(label)) {
    if (length(label) != length(x)) stop("label length must match x length.")
    tick_lab <- as.character(label[sel])
  } else {
    # default: show feature indices (R: 1-based)
    tick_lab <- as.character(sel)
  }

  df <- data.frame(k = seq_len(max_comp), value = vals)

  p <- ggplot(df, aes(x = k, y = value, group = 1)) +
    geom_line(linewidth = 0.8, color = point_color) +
    geom_point(size = 2.2, color = point_color) +
    scale_x_continuous(breaks = seq_len(max_comp), labels = tick_lab) +
    scale_y_continuous(expand = ggplot2::expansion(mult = c(0.02, 0.08))) +
    labs(title = title, x = xtitle, y = ytitle) +
    .vis_theme(rotation = rotation)

  if (!is.null(save_name)) {
    ggsave(filename = paste0(save_name, ".pdf"), plot = p, width = 7, height = 4)
  }

  print(p)
  invisible(sel)
}

# Cumulative explained variance plot
vis$cumu <- function(
  x,
  xtitle = "Top-k features",
  ytitle = "Cumulative explained variance",
  title = "Cumulative explained variance by top features",
  max_comp = 10,
  label = NULL,
  rotation = 0,
  label_size = 3,
  main_color = "black",
  save_name = NULL
) {
  x <- as.numeric(x)
  r2 <- sum(x, na.rm = TRUE)

  max_comp <- min(as.integer(max_comp), length(x))
  ord <- order(x, decreasing = TRUE)
  sel <- ord[seq_len(max_comp)]
  vals <- x[sel]
  cumu <- cumsum(vals)

  # include a (0,0) start point for the cumulative curve (but don't label 0 on x-axis)
  df <- data.frame(k = c(0, seq_len(max_comp)), cumu = c(0, cumu))

  # optional labels for selected features (must match length of x)
  if (!is.null(label)) {
    if (length(label) != length(x)) stop("label length must match x length.")
    tick_lab <- as.character(label[sel])
  } else {
    # default: show feature indices (R: 1-based)
    tick_lab <- as.character(sel)
  }
  # per-step increases between points (k-1 -> k)
  df_step <- data.frame(
    k = seq_len(max_comp),
    y0 = c(0, cumu[seq_len(max_comp - 1L)]),
    y1 = cumu,
    inc = vals,
    feat = tick_lab
  )
  df_step_show <- df_step[is.finite(df_step$inc), , drop = FALSE]
  label_scale <- max(abs(c(0, r2, cumu)), .Machine$double.eps)

  # Keep every max_comp selection. Long feature names wrap only at an
  # underscore, then each feature/value pair stays fixed below its point.
  wrap_feature <- function(feature, width = 14L) {
    if (nchar(feature) <= width || !grepl("_", feature, fixed = TRUE)) {
      return(feature)
    }
    parts <- strsplit(feature, "_", fixed = TRUE)[[1L]]
    if (length(parts) < 2L) return(feature)
    split_at <- seq_len(length(parts) - 1L)
    line_width <- vapply(split_at, function(i) {
      max(
        nchar(paste(parts[seq_len(i)], collapse = "_")),
        nchar(paste(parts[(i + 1L):length(parts)], collapse = "_"))
      )
    }, numeric(1L))
    i <- split_at[which.min(line_width)]
    paste(
      paste(parts[seq_len(i)], collapse = "_"),
      paste(parts[(i + 1L):length(parts)], collapse = "_"),
      sep = "\n"
    )
  }
  wrapped_feature <- vapply(df_step_show$feat, wrap_feature, character(1L))
  df_step_show$lab <- paste0(
    wrapped_feature,
    "\n+",
    formatC(df_step_show$inc, format = "f", digits = 3)
  )
  df_step_show$label_y <- df_step_show$y1 - 0.026 * label_scale
  total_label <- deparse(
    bquote("Total " * R^2 * ": " * .(formatC(r2, format = "f", digits = 3)))
  )

  p <- ggplot(df, aes(x = k, y = cumu, group = 1)) +
    # Per-feature increment from the previous cumulative value to this point.
    geom_segment(
      data = df_step_show,
      aes(x = k, xend = k, y = y0, yend = y1),
      inherit.aes = FALSE,
      arrow = grid::arrow(length = grid::unit(0.18, "cm")),
      linetype = "dashed",
      linewidth = 0.7,
      color = "grey60",
      alpha = 0.95
    ) +
    # Feature name and contribution form one compact label below the point.
    geom_text(
      data = df_step_show,
      aes(x = k, y = label_y, label = lab),
      inherit.aes = FALSE,
      hjust = 0.5,
      vjust = 1,
      size = label_size,
      family = .vis_font_family,
      color = "grey35",
      lineheight = 0.9,
      check_overlap = FALSE
    ) +
    geom_line(linewidth = 0.9, color = main_color) +
    geom_point(size = 2.4, color = main_color) +
    # total R^2 reference (dashed) + label
    geom_hline(yintercept = r2, linetype = "dashed", linewidth = 0.7, color = "grey40") +
    annotate(
      "text",
      x = 0.15,
      y = r2,
      label = total_label,
      hjust = 0,
      vjust = -0.6,
      size = 4,
      family = .vis_font_family,
      fontface = "bold",
      parse = TRUE
    ) +
    scale_x_continuous(
      breaks = seq_len(max_comp),
      labels = as.character(seq_len(max_comp)),
      limits = c(0, max_comp + 1),
      expand = ggplot2::expansion(mult = c(0.01, 0.02))
    ) +
    scale_y_continuous(expand = ggplot2::expansion(mult = c(0.02, 0.18))) +
    labs(title = title, x = xtitle, y = ytitle) +
    .vis_theme(
      rotation = rotation,
      plot_margin = ggplot2::margin(t = 10, r = 18, b = 12, l = 14)
    )

  if (!is.null(save_name)) {
    ggsave(filename = paste0(save_name, ".pdf"), plot = p, width = 8.5, height = 5.2)
  }

  print(p)
  invisible(p)
}

# Generalized correlation = sqrt(rsq contributions)
vis$gcorr <- function(
  x,
  color_map_name = "Blues",
  horizontal = FALSE,
  max_feature = 10,
  cutoff = 0,
  title = "Generalized correlation of features to the outcome",
  xtitle = "Feature",
  ytitle = "Generalized correlation",
  rotation = 0,
  label = NULL,
  decimal = 3,
  save_name = NULL
) {
  vis$rsq(
    sqrt(pmax(0, as.numeric(x))),
    color_map_name = color_map_name,
    horizontal = horizontal,
    model_rsq = FALSE,
    max_feature = max_feature,
    cutoff = cutoff,
    title = title,
    xtitle = xtitle,
    ytitle = ytitle,
    rotation = rotation,
    label = label,
    decimal = decimal,
    save_name = save_name
  )
}

# Histogram of Shapley R^2 contributions (distribution)
vis$hist <- function(
  x,
  bins = 30,
  title = expression(bold(paste("Distribution of Shapley ", R^2, " contributions"))),
  xtitle = expression(paste("Shapley ", R^2, " contribution")),
  ytitle = "Density",
  trim_nonfinite = TRUE,
  show_density = TRUE,
  density_adjust = 1,
  rotation = 0,
  main_color = "black",
  fill_color = "grey80",
  alpha = 0.85,
  save_name = NULL
) {
  x <- as.numeric(x)
  if (isTRUE(trim_nonfinite)) x <- x[is.finite(x)]
  if (length(x) == 0L) stop("No finite values to plot.")

  df <- data.frame(value = x)

  p <- ggplot(df, aes(x = value)) +
    geom_histogram(
      aes(y = after_stat(density)),
      bins = as.integer(bins),
      color = main_color,
      fill = fill_color,
      alpha = alpha
    ) +
    { if (isTRUE(show_density)) geom_density(adjust = density_adjust, linewidth = 0.9, color = main_color) else NULL } +
    labs(title = title, x = xtitle, y = ytitle) +
    .vis_theme(rotation = rotation)

  if (!is.null(save_name)) {
    ggsave(filename = paste0(save_name, ".pdf"), plot = p, width = 7, height = 4.2)
  }

  print(p)
  invisible(p)
}

# Density-only plot of Shapley R^2 contributions
vis$density <- function(
  x,
  title = expression(bold(paste("Density of Shapley ", R^2, " contributions"))),
  xtitle = expression(paste("Shapley ", R^2, " contribution")),
  ytitle = "Density",
  trim_nonfinite = TRUE,
  density_adjust = 1,
  rotation = 0,
  main_color = "black",
  save_name = NULL
) {
  x <- as.numeric(x)
  if (isTRUE(trim_nonfinite)) x <- x[is.finite(x)]
  if (length(x) == 0L) stop("No finite values to plot.")

  df <- data.frame(value = x)

  p <- ggplot(df, aes(x = value)) +
    geom_density(adjust = density_adjust, linewidth = 0.9, color = main_color) +
    labs(title = title, x = xtitle, y = ytitle) +
    .vis_theme(rotation = rotation)

  if (!is.null(save_name)) {
    ggsave(filename = paste0(save_name, ".pdf"), plot = p, width = 7, height = 4.2)
  }

  print(p)
  invisible(p)
}


#' Plot Q-SHAP R-squared contributions
#'
#' Convenience wrapper that works for both a `qshap_rsq` object and a plain
#' numeric vector of contributions. Use this if you have a numeric vector and
#' still want to pass arguments like `color_map_name`.
#'
#' @param x A `qshap_rsq` object (recommended) or a numeric vector.
#' @param type Plot type; see `plot.qshap_rsq`. Use `"loss"` to launch the
#'   interactive explorer or `"heatmap"` for the static raster heatmap.
#' @param ... Additional arguments passed to the underlying visualization
#'   function (e.g., `label`, `rotation`, `color_map_name`, `max_feature`).
#'
#' @return The ggplot2 plot object (invisibly)
#'
#' @examples
#' library(xgboost)
#' set.seed(42)
#' n <- 100
#' p <- 100
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' y <- X[, 1] - X[, 2] + rnorm(n, sd = 0.2)
#' model <- xgboost(X, y, nrounds = 15L, max_depth = 2L, verbosity = 0L, nthreads = 1L)
#' explainer <- gazer(model)
#' phi_rsq <- rsq(explainer, X, y)
#' plot(phi_rsq)
#'
#' @keywords internal
plot_qshap <- function(x, type = c("rsq", "elbow", "cumu", "gcorr", "hist", "density", "loss", "heatmap", "loss_heatmap"), ...) {
  # If x is a qshap_rsq object, reuse the S3 method
  if (inherits(x, "qshap_rsq")) {
    return(plot(x, type = type, ...))
  }

  # Otherwise treat x as a numeric vector and call vis functions directly
  type <- match.arg(type)

  # Interactive loss explorer (Shiny)
  # - If x is a qshap_rsq object with $loss, use x$loss
  # - If x is already a loss matrix/array (n x p), use it directly
  if (identical(type, "loss")) {
    if (inherits(x, "qshap_rsq") && is.list(x) && !is.null(x$loss)) {
      return(invisible(vis$loss(x$loss, ...)))
    }
    if (is.matrix(x) || is.array(x)) {
      return(invisible(vis$loss(x, ...)))
    }
    stop("type='loss' expects a qshap_rsq object with $loss, or a loss matrix/array (n x p). If you used qshap_rsq(..., local=TRUE), pass rsq_cons[[2]] (the loss matrix).")
  }

  if (type %in% c("heatmap", "loss_heatmap")) {
    if (is.matrix(x) || is.data.frame(x) || is.array(x)) {
      return(invisible(plot_loss_heatmap(x, ...)))
    }
    stop("type='heatmap' expects a qshap_rsq object with $loss, or a loss matrix/data frame (n x p).")
  }

  rsq_values <- as.numeric(x)

  invisible(
    switch(type,
      rsq = vis$rsq(rsq_values, ...),
      elbow = vis$elbow(rsq_values, ...),
      cumu = vis$cumu(rsq_values, ...),
      gcorr = vis$gcorr(rsq_values, ...),
      hist = vis$hist(rsq_values, ...),
      density = vis$density(rsq_values, ...)
    )
  )
}
