# makefile for latex files

# choose compiler 
CC = pdflatex 
CCTEST = pdflatex -draftmode

# source files ( no .tex extension!!!)
SRC = generations_psteger
BIB = $(generations_psteger.bib)

all: testtex pdf
# all: eps tex pdf
# plain latex compilation, produces a .dvi

testtex:
#testbib
	$(CCTEST) $(SRC)
	$(CCTEST) $(SRC)

testbib: $(BIB)
	$(CCTEST) $(SRC)
	bibtex $(SRC)

tex:
# bib
	$(CC) $(SRC)
	$(CC) $(SRC)

# first generate .aux, then compile library file
bib:	$(BIB)
	$(CC) $(SRC)
	bibtex $(SRC)

# make a ps file
ps:	tex
	dvips -o $(SRC).ps -t a4 $(SRC).dvi

# make a pdf file
pdf:	tex
#	dvips -Ppdf -G0  -o $(SRC).ps $(SRC).dvi
#	ps2pdf -dPDFsettings=/prepress $(SRC).ps

# prepare images for inclusion via eps
eps:	
	for i in fig/*.png; do echo $i; done

# rm the tex crappy files and other twiggle files
clean:
	rm -rf *.aux *.dvi *.log *.toc *.lof *.lot *.blg *.bbl *.end *~

# rm ps and pdf files as well
tidy: clean
	rm -rf $(SRC).ps $(SRC).pdf
