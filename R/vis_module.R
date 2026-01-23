#' Visualization module
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
    sorted_label <- as.character(ord - 1L)  # mimic python: feature index
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

  pal <- .vis_palette(color_map_name, n = 256)
  df$fill <- pal[pmax(1, pmin(256, 1 + floor(df$val_norm * 255)))]

  # label text
  df$txt <- formatC(df$value, format = "f", digits = decimal)

  # base plot no grid

p <- ggplot(df, aes(x = feature, y = value)) +
    geom_col(aes(fill = fill), width = 0.8, show.legend = FALSE) +
    scale_fill_identity() +
    labs(title = title, x = xtitle, y = ytitle) +
    theme_minimal(base_size = 12) +
    theme(
        axis.text.x = element_text(angle = rotation, hjust = 1, vjust = 1),
         panel.grid = element_blank(),
         axis.line = element_line(color = "black")
    )

  # add value labels
  # (offset like python: 2% max)
  offset <- max(df$value, na.rm = TRUE) * 0.02
p <- ggplot(df, aes(x = feature, y = value)) +
    geom_col(aes(fill = fill), width = 0.8, show.legend = FALSE) +
    scale_fill_identity() +
    labs(title = title, x = xtitle, y = ytitle) +
    theme_minimal(base_size = 12) +
    theme(
        panel.grid = element_blank(),
        axis.line = element_line(color = "black"),
        axis.text.x = element_text(angle = if (!horizontal) rotation else 0, hjust = 1, vjust = 1),
        axis.text.y = element_text(angle = if (horizontal) rotation else 0, hjust = 1, vjust = 0.5)
    )

    if (horizontal) p <- p + coord_flip()

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
    ggsave(filename = paste0(save_name, ".pdf"), plot = p, width = 7, height = 4)
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
  xtitle = "Feature Index",
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
  xtitle = "Feature Number",
  ytitle = "Explained Variance",
  max_comp = 10,
  title = "Explained Variance by Top Features"
) {
  x <- as.numeric(x)
  max_comp <- min(as.integer(max_comp), length(x))

  ord <- order(x, decreasing = TRUE)
  sel <- ord[seq_len(max_comp)]
  vals <- x[sel]

  df <- data.frame(k = seq_len(max_comp), value = vals)

p <- ggplot(df, aes(x = k, y = value)) +
    geom_line(linetype = "dashed") +
    geom_point() +
    scale_x_continuous(breaks = seq_len(max_comp)) +
    labs(title = title, x = xtitle, y = ytitle) +
    theme_minimal(base_size = 12) +
    theme(
         panel.grid = element_blank(),
        axis.line = element_line(color = "black")
    )

  print(p)
  invisible(sel)
}

# Cumulative explained variance plot
vis$cumu <- function(
  x,
  xtitle = "Feature Number",
  ytitle = "Cumulative Explained Variance",
  title = "Cumulative Explained Variance by Top Features",
  max_comp = 10,
  save_name = NULL
) {
  x <- as.numeric(x)
  r2 <- sum(x, na.rm = TRUE)

  max_comp <- min(as.integer(max_comp), length(x))
  ord <- order(x, decreasing = TRUE)
  vals <- x[ord][seq_len(max_comp)]
  cumu <- cumsum(vals)

  df <- data.frame(k = seq_len(max_comp), cumu = cumu)

p <- ggplot(df, aes(x = k, y = cumu)) +
    geom_line() +
    geom_point() +
    geom_hline(yintercept = r2, linetype = "dashed") +
    scale_x_continuous(breaks = seq_len(max_comp)) +
    labs(title = title, x = xtitle, y = ytitle) +
    theme_minimal(base_size = 12) +
    theme(
            panel.grid = element_blank(),
            axis.line = element_line(color = "black")
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
  xtitle = "Feature index",
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