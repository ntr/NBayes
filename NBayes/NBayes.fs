module NBayes

open Deedle
open System

let sample ratio (frame : Frame<_, _>) =
                                         
    let numbersToPick = (float frame.RowCount) * ratio |> int
    
    let indicesToPick = 
        [ 0..frame.RowCount - 1 ]
        |> List.sortBy (fun _ -> System.Guid.NewGuid())
        |> Seq.take numbersToPick
        |> Seq.toArray
    frame.GetRowsAt indicesToPick

let split leftRatio (frame : Frame<_, _>) = 
    let numbersToPick = (float frame.RowCount) * leftRatio
                        |> int
    let indices = [| 0..frame.RowCount - 1 |] |> Array.sortBy (fun _ -> System.Guid.NewGuid())
    (frame.GetRowsAt indices.[..numbersToPick], frame.GetRowsAt indices.[numbersToPick + 1..])

let getColumnAt index (frame : Frame<_, _>) = frame.GetColumnAt index

let binaryClassification frame actual predicted = 
    let standartizeAttribute value =
        value.ToString().ToLower()
        
    frame 
    |> Frame.pivotTable 
        (fun _ r -> r.Get(actual))
        (fun _ r -> r.Get(predicted))    
        (fun frame -> frame?Survived |> Stats.count)
    |> Frame.mapColKeys (standartizeAttribute>>(sprintf "Predicted %s"))
    |> Frame.mapRowKeys (standartizeAttribute>>(sprintf "Actual %s"))

let normalDistrubutionFun mean stdev = 
    let inline sqr x = x * x
    (fun x -> 
    let p = -sqr (x - mean) / (2. * sqr (stdev))
    Math.Exp(p) / (stdev * sqrt (2. * Math.PI)))

type DataModel = {NumericAttributes: string list; NominalAttributes:string list; TargetAttribute:string}

let classify model (frame:Frame<int,string>) = 
    let getClassifier {NumericAttributes=numericAttributes; NominalAttributes=nominalAttributes; TargetAttribute=targetAttribute} data = 
        let targetAttributeStats = 
            let survivedGroups = 
                data
                |> Frame.groupRowsByString targetAttribute
                |> getColumnAt 0
                |> Stats.levelCount fst
                |> Series.mapValues float
        
            let result = Frame([ "Count" ], [ survivedGroups ])
            result?PriorProb <- result?Count / (float data.RowCount)
            result
    
        let nominalLikehoodsLookups = 
            let makeLikehoodLookup attribute frame = 
                frame
                |> Frame.groupRowsBy attribute
                |> Frame.groupRowsByString targetAttribute
                |> Frame.mapRowKeys Pair.flatten3
                |> Frame.getCol targetAttribute
                |> Stats.levelCount Pair.get1And2Of3 //counts groupped by attribute and target attribute
                |> Series.map (fun (survItem, _) value -> (float value) / targetAttributeStats?Count.[survItem]) // converting counts to ratios
                |> Series.mapKeys (fun (f, s) -> (attribute, f, s)) 
            nominalAttributes
            |> List.map (fun attr -> makeLikehoodLookup attr data)
            |> Series.mergeAll
    
        let targetAttributeValues = 
            data.GetColumn<string> targetAttribute
            |> Series.values 
            |> Seq.distinct 
            |> Seq.toList
    
        let numericLikehoodCalculator = //only normal distribution is supported
            let predictors = 
                [ for attr in numericAttributes do
                    for value in targetAttributeValues do
                        yield (attr, value), 
                            let column = (data |> Frame.filterRowValues (fun row -> row.GetAs<string>(targetAttribute) = value) |> Frame.getCol attr)
                            normalDistrubutionFun (column |> Stats.mean) (column |> Stats.stdDev)] 
                |> Map.ofList
            fun name value targetValue -> 
                match value with
                | OptionalValue.Present a -> predictors.[(name, targetValue)] a
                | OptionalValue.Missing -> 1.

        fun (row : ObjectSeries<string>) ->
            let aggregate attributes f = 
                attributes
                |> List.map f
                |> List.reduce (*)

            let result = 
                targetAttributeValues
                |> List.map 
                    (fun targetValue -> 
                        (targetValue, targetAttributeStats?PriorProb.[targetValue]
                                 * (aggregate nominalAttributes (fun attribute -> nominalLikehoodsLookups.[(attribute, targetValue, row.Get(attribute))])) 
                                 * (aggregate numericAttributes (fun attribute -> numericLikehoodCalculator attribute (row.TryGetAs<float>(attribute)) targetValue) ) )) 

            ("Value", result |> List.maxBy snd |> fst :> obj) :: (result |> List.map(fun (a,b) -> (a+"Prob",box b))) |> series      
   
    let classifier = getClassifier model frame
    frame?Predicted <- Frame.mapRowValues classifier frame
    frame |> Frame.expandCols([|"Predicted"|])