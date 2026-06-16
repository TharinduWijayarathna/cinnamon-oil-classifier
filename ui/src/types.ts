export type PredictionResponse = {
  prediction: string;
  predicted_class_index: number;
  confidence_percentage: number;
  raw_prediction_values: number[] | number[][];
  classification_type: "binary" | "multiclass";
};

export type ValidationErrors = {
  image: boolean;
  density: boolean;
  ph: boolean;
};
