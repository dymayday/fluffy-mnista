extern crate rayon;
extern crate fluffy_penguin;
extern crate mnist;
extern crate rulinalg;

use rayon::prelude::*;
use fluffy_penguin::genetic_algorithm::individual::Specimen;
use fluffy_penguin::genetic_algorithm::Population;

const ROWS: usize = 28;
const COLS: usize = 28;
const DATASET_SIZE: usize = 1_000;

fn load_dataset() -> (Vec<f32>, Vec<f32>) {

    use mnist::{Mnist, MnistBuilder};

    let (trn_size, rows, cols) = (DATASET_SIZE, ROWS, COLS);

    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .base_path("/home/meidhy/work/data/mnist/")
        // .label_format_digit()
        .label_format_one_hot()
        .training_set_length(trn_size as u32)
        .validation_set_length(100)
        .test_set_length(100)
        .finalize();


    for i in 0..10 {
        print!("{:>2}", i)
    }
    println!();
    for i in 0..10 {
        print!("{:>2}", trn_lbl[i])
    }

    for i in 0..rows*cols {
            if i % cols == 0 {
                println!();
            }
            print!("{:>4}", trn_img[i]);
    }
    println!();

    let trn_img: Vec<f32> = trn_img.iter().map(|x| *x as f32).collect();
    let trn_lbl: Vec<f32> = trn_lbl.iter().map(|x| *x as f32).collect();

    (trn_img, trn_lbl)
}


/// test specimen on 100 data point, and return the mean squared
/// error of the result
fn compute_specimen_score(specimen: &Specimen<f32>, trn_img: &Vec<f32>, trn_lbl: &Vec<f32>) -> f32 {
    let mut specimen = specimen.clone();

    let dataset_size: usize = DATASET_SIZE;
    // compute the squared difference between the spcimen ANN output and the model to fit
    // for each data point
    let mut squared_errors: Vec<f32> = Vec::with_capacity(dataset_size * 10);

    let mut i: usize = 0;
    while i < DATASET_SIZE {

        let inputs = &trn_img[i..i+(ROWS*COLS)];
        specimen.update_input(&inputs);
        let specimen_output = specimen.evaluate();

        let model_output: Vec<f32> = trn_lbl[i*10..i*10+10].iter().map(|x| *x).collect();

        for e in 0..10 {
            squared_errors.push( (model_output[e] - specimen_output[e]).powf(2.0) );
        }
        i += ROWS * COLS;
    }

    // return the RMSE
    let error_sum = squared_errors.iter().fold(0., |sum, err| sum + err);
    error_sum / (dataset_size as f32)
}



fn train_model(trn_img: Vec<f32>, trn_lbl: Vec<f32>) {
    use std::cmp::Ordering;

    let population_size: usize = 100;
    let input_size: usize = ROWS * COLS;
    let output_size: usize = 10;
    let mutation_probability: f32 = 0.10;

    let mut generation_counter: usize = 0;
    let cycle_per_structure: usize = 1000;
    let cycle_stop: usize = 10000;

    let mut population: Population<f32> = Population::new(
        population_size,
        input_size,
        output_size,
        mutation_probability,
        );

    population.exploration();
    for _ in 0..cycle_stop {
        generation_counter += 1;

        let scores: Vec<f32> = population.species.par_iter()
            .map(|specimen| compute_specimen_score(specimen, &trn_img, &trn_lbl))
            .collect();

        // Update fitness of each specimen.
        // High score needs to represent a better fitness.
        for i in 0..population_size {
            population.species[i].fitness = -scores[i];
        }

        // Selection phase.
        population.evolve();

        // Lookup for some better weights.
        if generation_counter % cycle_per_structure == 0 {
            population.exploration();
        } else {
            population.exploitation();
        }


        let best_score = scores
            .iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Greater))
            .unwrap();
        let mean_score: f32 = scores.iter().sum::<f32>() / population_size as f32;
        println!(
            "[{:>5}], best RMSE = {:.6} , mean = {:.6}",
            generation_counter, best_score, mean_score
        );
    }

}


fn main() {
    let (trn_img, trn_lbl): (Vec<f32>, Vec<f32>) = load_dataset();
    train_model(trn_img, trn_lbl);
}
