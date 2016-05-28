(TeX-add-style-hook "header"
 (lambda ()
    (TeX-add-symbols
     '("TODO" 1)
     '("changefont" 3)
     '("bcite" 1))
    (TeX-run-style-hooks
     "aas_macros"
     "amsmath"
     "reqno"
     "namelimits"
     "sumlimits"
     "amssymb"
     "color"
     "graphicx"
     "pdftex"
     "natbib"
     ""
     "inputenc"
     "latin1"
     "fontenc"
     "T1"
     "latex2e"
     "art10"
     "article"
     "a4paper")))

