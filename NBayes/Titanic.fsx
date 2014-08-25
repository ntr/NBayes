#nowarn "211"
#I "../../bin"
#I "../bin"
#I "bin"
#I "lib"
#I "../packages/Deedle.1.0.2/lib/net40"
// Also reference path with FSharp.Data.DesignTime.dll
#I "../packages/FSharp.Data.2.0.8/lib/net40/"
// Reference Deedle
#r "Deedle.dll"
#load "NBayes.fs"
open NBayes

do fsi.AddPrinter(fun (printer : Deedle.Internal.IFsiFormattable) -> "\n" + (printer.Format()))

open Deedle
open System

Environment.CurrentDirectory <- __SOURCE_DIRECTORY__
let titanic = Frame.ReadCsv("data/titanic.csv", inferTypes = true)

let classifed = classify {NominalAttributes=["Pclass"; "Sex"]; NumericAttributes=["Age";]; TargetAttribute= "Survived"} titanic
let binary = binaryClassification classifed "Survived" "Predicted.Value"
let ratio = (binary.["Predicted true"].["Actual true"] + binary.["Predicted false"].["Actual false"]) / (float classifed.RowCount)
