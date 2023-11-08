
import sys
import re
import shutil
import os
import magic

cppcommentpat = re.compile("(^\/\/)")
MLIRcommentpat = re.compile("^\/\/.?(?!(RUN)|(REQUIRES)|(CHECK)).*")
shellcommentpat = re.compile("(^#)")
shebangpat = re.compile("(^#!)")
whitespacepat = re.compile("(^\s*$)")
copyrightpat = re.compile("Copyright (\d*)(?:-(\d*))? Xilinx")

#//===- BasicpyOps.cpp - Core numpy dialect ops -------------------*- C++-*-===//

header = '''
 This file is licensed under the MIT License.
 SPDX-License-Identifier: MIT
 
'''

def genheaderline(startline, midline, endline):
  l = len(startline) + len(endline)
  return startline + (midline*(80-l)) + endline + "\n"

def outputwith(output, prefix, header):
  for line in header.splitlines():
    output(prefix + line + "\n")

def gencopyrightline(prefix, copy):
  (created, current) = copy
  if(created == current):
      return prefix + " (c) Copyright " + created + " Advanced Micro Devices, Inc.\n"

  return prefix + " (c) Copyright " + created + "-" + current + " Advanced Micro Devices, Inc.\n"

def getCopyright(fname):
  f = open(fname, "r")
  copy = ("2023", "2023")
  for x in f:
    m = cppcommentpat.match(x)
    if m:
      m = copyrightpat.search(x)
      if m:
        if m.lastindex == 1:
          copy = (m.group(1, 1))
          break
        elif m.lastindex == 2: 
          copy = (m.group(1, 2))
          break
  f.close()
  return copy

def getshebang(fname):
  f = open(fname, "r")
  for x in f:
    m = shebangpat.match(x)
    if m:
      f.close()
      return x
    else:
      f.close()
      return None

for fname in sys.argv[1:]:
  f = open(fname, "r")
  fout = open(fname + ".new", "w")

  output = fout.write
  name = os.path.basename(fname)
  (root, ext) = os.path.splitext(name)

  if ext == ".txt":
    filetype = "cmake"
    outputwith(output, "#", header)
    output(gencopyrightline("#", getCopyright(fname)))
    output("\n")
    commentpat = shellcommentpat
  elif ext == ".py":
    filetype = "python"
    bang = getshebang(fname)
    if bang:
      output(bang)
    output("# " + fname + " -*- Python -*-\n")
    outputwith(output, "#", header)
    output(gencopyrightline("#", getCopyright(fname)))
    output("\n")
    commentpat = shellcommentpat
  elif ext == ".td":
    filetype = "tablegen"
    output(genheaderline("//===- " + name + " ", "-", "*- " + filetype + " -*-===//"))
    outputwith(output, "//", header)
    output(gencopyrightline("//", getCopyright(fname)))
    output("//\n")
    output(genheaderline("//===-", "-", "-===//"))
    output("\n")
    commentpat = cppcommentpat
  elif ext == ".cpp" or ext == ".cc" or ext == ".h":
    getCopyright(fname)
    filetype = "C++"
    output(genheaderline("//===- " + name + " ", "-", "*- " + filetype + " -*-===//"))
    outputwith(output, "//", header)
    output(gencopyrightline("//", getCopyright(fname)))
    output("//\n")
    output(genheaderline("//===-", "-", "-===//"))
    output("\n")
    commentpat = cppcommentpat
  elif ext == ".mlir":
    getCopyright(fname)
    filetype = "MLIR"
    output(genheaderline("//===- " + name + " ", "-", "*- " + filetype + " -*-===//"))
    outputwith(output, "//", header)
    output(gencopyrightline("//", getCopyright(fname)))
    output("//\n")
    output(genheaderline("//===-", "-", "-===//"))
    output("\n")
    commentpat = MLIRcommentpat
  elif re.compile("ASCII text executable").search(str(magic.from_file(fname))):
    # shell script?
    bang = getshebang(fname)
    if bang:
      output(bang)
    outputwith(output, "#", header)
    output(gencopyrightline("#", getCopyright(fname)))
    output("\n")
    commentpat = shellcommentpat

  else:
    print("Unrecognized file: " + fname + " " + magic.from_file(fname))    
    os.remove(fname + ".new")
    continue
    
  headerDone = False
  whitespaceDone = False
  for x in f:
    if not commentpat.match(x):
      headerDone = True
    if headerDone:
      if not whitespacepat.match(x):
        whitespaceDone = True
      if whitespaceDone:
        output(x)
    # else:
    #   print("header:",x)

  shutil.move(fname + ".new", fname)
  f.close()