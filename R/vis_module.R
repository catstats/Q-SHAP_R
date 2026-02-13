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
#'   \item{loss}{Interactive loss explorer (requires shiny)}
#' }
#' 
#' @export
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
#' @param type Plot type: one of "rsq", "elbow", "cumu", "gcorr", or "loss".
#' @param ... Passed to the underlying visualization function.
#'
#' @return A ggplot2 object (invisibly).
#'
#' @method plot qshap_rsq
#' @export
plot.qshap_rsq <- function(x, y = NULL,
                          type = c("rsq", "elbow", "cumu", "gcorr", "loss"), ...) {
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

  if (is.null(rsq_values)) {
    stop("Cannot find R² contributions in this qshap_rsq object. Expected one of: $rsq, $phi_rsq, $contrib.")
  }

  rsq_values <- as.numeric(rsq_values)

  invisible(
    switch(type,
           rsq = vis$rsq(rsq_values, ...),
           elbow = vis$elbow(rsq_values, ...),
           cumu = vis$cumu(rsq_values, ...),
           gcorr = vis$gcorr(rsq_values, ...))
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
  invisible(x)
}

#' Summary method for qshap_rsq objects
#'
#' Provides a summary of the qshap_rsq object, showing the top features by R-squared contribution
#'
#' @param object A \code{qshap_rsq} object
#' @param n Integer number of top features to display (default: 10)
#' @param ... Additional arguments (currently unused)
#' @export
summary.qshap_rsq <- function(object, n = 10, ...) {
  cat("Q-SHAP R² Summary\n")
  cat("=================\n\n")
  
  if (is.null(object$rsq)) {
    cat("No R² values available in this object.\n")
    return(invisible(object))
  }
  
  rsq <- object$rsq
  total_rsq <- sum(rsq, na.rm = TRUE)
  
  cat("Overall Statistics:\n")
  cat("  Total R²:", round(total_rsq, 6), "\n")
  cat("  Number of features:", length(rsq), "\n")
  
  # Add SD and CI information if available
  if (!is.null(object$sd_rsq)) {
    cat("  Standard errors: Available\n")
  }
  if (!is.null(object$ci_lower) && !is.null(object$ci_upper)) {
    cat("  Confidence intervals:", paste0(round(object$level * 100, 1), "%"), "\n")
  }
  if (!is.null(object$loss)) {
    cat("  Loss matrix dim:", paste(dim(object$loss), collapse = " x "), "\n")
  }
  
  cat("\nR² Distribution:\n")
  cat("  Min:", round(min(rsq, na.rm = TRUE), 6), "\n")
  cat("  Q1:", round(quantile(rsq, 0.25, na.rm = TRUE), 6), "\n")
  cat("  Median:", round(median(rsq, na.rm = TRUE), 6), "\n")
  cat("  Mean:", round(mean(rsq, na.rm = TRUE), 6), "\n")
  cat("  Q3:", round(quantile(rsq, 0.75, na.rm = TRUE), 6), "\n")
  cat("  Max:", round(max(rsq, na.rm = TRUE), 6), "\n")
  
  # Count significant features
  sig_features <- sum(rsq > 0.01, na.rm = TRUE)
  cat("\nSignificant Features (R² > 0.01):", sig_features, "\n")
  
  # Show top N features
  cat("\nTop", min(n, length(rsq)), "features by R²:\n")
  
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
  
  # Add SD and CI columns if available
  if (!is.null(object$sd_rsq)) {
    df$SE <- object$sd_rsq
  }
  if (!is.null(object$ci_lower) && !is.null(object$ci_upper)) {
    df$CI_lower <- object$ci_lower
    df$CI_upper <- object$ci_upper
  }
  
  df <- df[order(df$R_squared, decreasing = TRUE), ]
  df <- head(df, n)
  
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

# Bar plot for Shapley R^2 (or any 1d contribution vector)
vis$rsq <- function(
  x,
  color_map_name = "Blues",
  horizontal = FALSE,
  model_rsq = TRUE,
  max_feature = 10,
  cutoff = 0,
  title = "Shapley R²",
  xtitle = "Feature index",
  ytitle = "R²",
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
        fontface = "plain"
      )
    } else {
      # after coord_flip(), x/y swap; use hjust to place label to the right of the bar
      p_label_layer <- geom_text(
        aes(label = txt),
        hjust = -0.15,
        size = 3.6,
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
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0),
      axis.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = if (!horizontal) rotation else 0, hjust = 1, vjust = 1),
      axis.text.y = element_text(angle = if (horizontal) rotation else 0, hjust = 1, vjust = 0.5),
      panel.grid.major.y = element_line(color = "grey85"),
      panel.grid.minor = element_blank()
    )

  if (horizontal) {
    p <- p + coord_flip(clip = "off")
  } else {
    # allow labels to extend slightly above the panel
    p <- p + coord_cartesian(clip = "off")
  }

  # add model rsq annotation
  if (model_rsq) {
    ann <- paste0("Model R²: ", formatC(x_sum, format = "f", digits = 3))
    p <- p + annotate("text",
      x = if (!horizontal) show_len else 1,
      y = max(df$value, na.rm = TRUE),
      label = ann,
      hjust = 1, vjust = 1,
      size = 4
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
  title = "Shapley Loss: Sample",
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
        shiny::numericInput("i", "Sample Index", value = 1, min = 1, max = nrow(loss), step = 1),
        shiny::checkboxInput("horizontal", "Horizontal", value = FALSE),
        shiny::numericInput("max_feature", "Max features", value = min(10, ncol(loss)), min = 1, max = ncol(loss), step = 1),
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

# Elbow plot: top contributions (sorted)
vis$elbow <- function(
  x,
  xtitle = "Top-k features",
  ytitle = "Explained Variance",
  max_comp = 10,
  title = "Explained Variance by Top Features",
  label = NULL,
  rotation = 0,
  point_color = "black"
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
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0),
      axis.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = rotation, hjust = 1, vjust = 1),
      panel.grid.major.y = element_line(color = "grey85"),
      panel.grid.minor = element_blank()
    )

  print(p)
  invisible(sel)
}

# Cumulative explained variance plot
vis$cumu <- function(
  x,
  xtitle = "Top-k features",
  ytitle = "Cumulative Explained Variance",
  title = "Cumulative Explained Variance by Top Features",
  max_comp = 10,
  label = NULL,
  rotation = 0,
  inc_threshold = 0.01,
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
  df_step$lab <- paste0(df_step$feat, ": +", formatC(df_step$inc, format = "f", digits = 3))
  df_step_show <- df_step[is.finite(df_step$inc) & (df_step$inc >= inc_threshold), , drop = FALSE]

  p <- ggplot(df, aes(x = k, y = cumu, group = 1)) +
    # per-feature increment: vertical arrowed dashed line (only for sufficiently large gains)
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
    # label to the right of the arrow, centered on the segment
    geom_text(
      data = df_step_show,
      aes(x = k + 0.18, y = (y0 + y1) / 2, label = lab),
      inherit.aes = FALSE,
      hjust = 0,
      vjust = 0.5,
      size = label_size,
      color = "grey35",
      check_overlap = TRUE
    ) +
    geom_line(linewidth = 0.9, color = main_color) +
    geom_point(size = 2.4, color = main_color) +
    # total R² reference (dashed) + label
    geom_hline(yintercept = r2, linetype = "dashed", linewidth = 0.7, color = "grey40") +
    geom_text(
      data = data.frame(x = max_comp, y = r2, lab = paste0("Total R²: ", formatC(r2, format = "f", digits = 3))),
      aes(x = x, y = y, label = lab),
      inherit.aes = FALSE,
      hjust = 1,
      vjust = -0.6,
      size = 4,
      fontface = "bold"
    ) +
    scale_x_continuous(
      breaks = seq_len(max_comp),
      labels = as.character(seq_len(max_comp)),
      limits = c(0, max_comp),
      expand = ggplot2::expansion(mult = c(0.01, 0.02))
    ) +
    scale_y_continuous(expand = ggplot2::expansion(mult = c(0.02, 0.08))) +
    labs(title = title, x = xtitle, y = ytitle) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0),
      axis.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = rotation, hjust = 1, vjust = 1),
      panel.grid.major.y = element_line(color = "grey85"),
      panel.grid.minor = element_blank()
    )

  if (!is.null(save_name)) {
    ggsave(filename = paste0(save_name, ".pdf"), plot = p, width = 7, height = 4)
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
  title = "Generalized Correlation of Features to the Outcome",
  xtitle = "Feature index (1-based)",
  ytitle = "Generalized Correlation",
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


#' Plot Q-SHAP R-squared contributions
#'
#' Convenience wrapper that works for both a `qshap_rsq` object and a plain
#' numeric vector of contributions. Use this if you have a numeric vector and
#' still want to pass arguments like `color_map_name`.
#'
#' @param x A `qshap_rsq` object (recommended) or a numeric vector.
#' @param type Plot type; see `plot.qshap_rsq`. Use `"loss"` to launch the interactive explorer (requires a loss matrix).
#' @param ... Additional arguments passed to the underlying visualization
#'   function (e.g., `label`, `rotation`, `color_map_name`, `max_feature`).
#'
#' @return The ggplot2 plot object (invisibly)
#'
#' @examples
#' \dontrun{
#' # Works on qshap_rsq objects
#' plot_qshap(phi_rsq, color_map_name = "viridis")
#'
#' # Works on numeric vectors
#' plot_qshap(rsq_contributions, color_map_name = "viridis")
#'
#' # Interactive loss explorer (requires local=TRUE)
#' # rsq_cons <- qshap_rsq(explainer, X, y, local = TRUE)
#' # plot_qshap(rsq_cons[[2]], type = "loss")
#' }
#'
#' @export
plot_qshap <- function(x, type = c("rsq", "elbow", "cumu", "gcorr", "loss"), ...) {
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

  rsq_values <- as.numeric(x)

  invisible(
    switch(type,
      rsq = vis$rsq(rsq_values, ...),
      elbow = vis$elbow(rsq_values, ...),
      cumu = vis$cumu(rsq_values, ...),
      gcorr = vis$gcorr(rsq_values, ...)
    )
  )
}

