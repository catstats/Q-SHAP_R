FROM python:3.13.18-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

LABEL org.opencontainers.image.source="https://github.com/catstats/Q-SHAP_R" \
      org.opencontainers.image.title="Q-SHAP replication environment" \
      org.opencontainers.image.version="3.0.0-py313"

ARG CRAN_MIRROR=https://cran.r-project.org
ARG MAKEFLAGS=-j2

ENV MPLBACKEND=Agg

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake curl fonts-dejavu-core libcurl4-openssl-dev libssl-dev libxml2-dev \
        r-base-core r-base-dev \
        r-cran-data.table r-cran-ggplot2 r-cran-httr r-cran-jsonlite r-cran-matrix \
        r-cran-progress r-cran-rcpp r-cran-rcppeigen r-cran-remotes \
        r-cran-stringi r-cran-xml \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir qshap==3.0.0 xgboost-cpu==3.4.1

RUN Rscript --vanilla -e 'options(repos = c(CRAN = Sys.getenv("CRAN_MIRROR")), timeout = 180); \
    remotes::install_version("xgboost", "3.1.3.1", upgrade = "never"); \
    remotes::install_version("lightgbm", "4.6.0", upgrade = "never"); \
    remotes::install_version("qshap", "3.0.0", upgrade = "never"); \
    install.packages(c("OpenML", "farff")); \
    stopifnot(packageVersion("qshap") == "3.0.0", \
              packageVersion("xgboost") == "3.1.3.1", \
              requireNamespace("OpenML", quietly = TRUE), \
              requireNamespace("farff", quietly = TRUE))'

RUN set -eu; \
    case "$(dpkg --print-architecture)" in \
        amd64) catboost_arch=x86_64 ;; \
        arm64) catboost_arch=aarch64 ;; \
        *) printf 'CatBoost R release binaries require amd64 or arm64.\n' >&2; exit 1 ;; \
    esac; \
    CATBOOST_URL="https://github.com/catboost/catboost/releases/download/v1.2.10/catboost-R-linux-${catboost_arch}-1.2.10.tgz" \
      Rscript --vanilla -e 'options(repos = c(CRAN = Sys.getenv("CRAN_MIRROR")), timeout = 180); \
        remotes::install_url(Sys.getenv("CATBOOST_URL"), upgrade = "never"); \
        stopifnot(packageVersion("catboost") == "1.2.10")'

# Cache the public California Housing data from OpenML (dataset 43939).
RUN mkdir -p /opt/qshap/openml-cache \
    && Rscript --vanilla -e 'cache <- "/opt/qshap/openml-cache"; \
        OpenML::setOMLConfig(server = "https://www.openml.org/api/v1", cachedir = cache); \
        housing <- OpenML::getOMLDataSet(data.id = 43939); \
        arff <- file.path(cache, "datasets", "43939", "dataset.arff"); \
        stopifnot(nrow(housing[["data"]]) == 20640L, \
                  unname(tools::md5sum(arff)) == "38cc9f7060e510f2f9f8072e250c5c35")'

RUN printf '\ninvisible(OpenML::setOMLConfig(server = "https://www.openml.org/api/v1", cachedir = "/opt/qshap/openml-cache"))\n' \
    >> /etc/R/Rprofile.site

ENV MPLCONFIGDIR=/tmp/qshap-matplotlib \
    NUMBA_CACHE_DIR=/tmp/qshap-numba

WORKDIR /work
CMD ["bash"]
