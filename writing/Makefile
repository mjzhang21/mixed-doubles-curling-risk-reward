source := $(wildcard *.tex)
source := $(filter-out setup.tex, $(source))  # Exclude setup.tex
bib := $(wildcard *.bib)
out := $(patsubst %.tex,%.pdf,$(source))
latest := $(shell git log -1 --pretty=format:"%h")
rev := $(latest)

.PHONY: all
all: $(out)

%.pdf: %.tex $(bib)
	@latexmk -halt-on-error -pdf $<
##	@latexmk -c

## compare modification with the latest version checked into git
## for comparison of specific commits:
##	latexdiff-vc --git -r old_githash -r new_githash --pdf source.tex
diff:
	latexdiff-vc --git --pdf --force $(source) -r $(rev)

log:
	@git log -5 --pretty=format:"%h by %an at %ar: %s" $(source)

.PHONY: clean
clean:
	rm -rf $(out) *~ .*~ .\#* .Rhistory *.aux *.bbl *.blg *.dvi *.out *.log *.toc \
	*.fff *.fdb_latexmk *.fls *.ttt *diff* *oldtmp* .blb
