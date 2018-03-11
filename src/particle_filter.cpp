/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
double INIT_PARTICLE_WEIGHT = 1.0;

using namespace std;

// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
//   x, y, theta and their uncertainties from GPS) and all weights to 1.
// Add random Gaussian noise to each particle.
// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
void ParticleFilter::init(double gps_x, double gps_y, double theta, double std[]) {
  num_particles = 100;
  particles.resize(num_particles);
  weights.resize(num_particles);

  default_random_engine gen;
  normal_distribution<double> dist_x(gps_x, std[0]);
  normal_distribution<double> dist_y(gps_y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for(int i = 0; i < num_particles; ++i) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = INIT_PARTICLE_WEIGHT;
  }

  is_initialized = true;
}

// TODO: Add measurements to each particle and add random Gaussian noise.
// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
//  http://www.cplusplus.com/reference/random/default_random_engine/
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for(int i = 0; i < num_particles; ++i) {
    particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta)) + dist_x(gen);
    particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t))) + dist_y(gen);
    particles[i].theta += (yaw_rate * delta_t) + dist_theta(gen);
  }
}

// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
//   observed measurement to this particular landmark.
// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
//   implement this method and use it as a helper during the updateWeights phase.
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> landmarks_visible_to_particle,
                                     std::vector<LandmarkObs>& trans_observations) {

  for(unsigned long i = 0; i < trans_observations.size(); ++i) {

    double current_min = 999999.0;
    unsigned int min_index = -1;

    for(unsigned long j = 0; j < landmarks_visible_to_particle.size(); ++j) {

      double current_dist = dist(trans_observations[i].x,
                                 trans_observations[i].y,
                                 landmarks_visible_to_particle[j].x,
                                 landmarks_visible_to_particle[j].y);

      if(current_dist < current_min) {
        current_min = current_dist;
        min_index = j;
      }
    }

    trans_observations[i].id = min_index;
  }
}

// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
//   The following is a good resource for the theory:
//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
//   and the following is a good resource for the actual equation to implement (look at equation
//   3.33
//   http://planning.cs.uiuc.edu/node99.html
void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

  // 1. Convert observations from Car Co-ordinates to Map Co-ordinates.
  std::vector<LandmarkObs> trans_observations;
  std::vector<LandmarkObs> landmarks_visible_to_particle;
  double gaussian_norm = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1]));

  for(unsigned int i = 0; i < particles.size(); ++i) {

    // Find out which landmarks are visible to particle
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      double _dist = dist(map_landmarks.landmark_list[j].x_f,
                          map_landmarks.landmark_list[j].y_f,
                          particles[i].x,
                          particles[i].y);

      if (_dist < sensor_range) {
        LandmarkObs landmark_within_range = {
          map_landmarks.landmark_list[j].id_i,
          map_landmarks.landmark_list[j].x_f,
          map_landmarks.landmark_list[j].y_f,
        };

        landmarks_visible_to_particle.push_back(landmark_within_range);
      }
    }

    // Convert what car sees (observations) to particle's frame of reference
    for(unsigned j = 0; j < observations.size(); ++j) {
      double trans_x = particles[i].x +
        (cos(particles[i].theta) * observations[j].x) -
        (sin(particles[i].theta) * observations[j].y);

      double trans_y = particles[i].y +
        (sin(particles[i].theta) * observations[j].x) -
        (cos(particles[i].theta) * observations[j].y);

      LandmarkObs trans_observation = {
        observations[j].id,
        trans_x,
        trans_y,
      };

      trans_observations.push_back(trans_observation);
    }

    // Associate closest landmark out of the ones that the car sees
    dataAssociation(landmarks_visible_to_particle, trans_observations);

    double particle_weight = particles[i].weight;
    for(unsigned int j = 0; j < trans_observations.size(); ++j) {
      LandmarkObs closest_landmark = landmarks_visible_to_particle[trans_observations[i].id];

      double x = pow((trans_observations[j].x - closest_landmark.x),
                     2);
      double y = pow((trans_observations[j].y - closest_landmark.y),
                     2);

      double exponent1 = (x / (2 * pow(std_landmark[0], 2)));
      double exponent2 = (y / (2 * pow(std_landmark[1], 2)));
      double exponent = (exponent1 + exponent2);
      double weight = gaussian_norm * exp(-exponent);
      particle_weight *= weight;
    }

    particles[i].weight = particle_weight;
    weights[i] = particle_weight;
  } // For each particle
}

// TODO: Resample particles with replacement with probability proportional to their weight.
// NOTE: You may find std::discrete_distribution helpful here.
//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
void ParticleFilter::resample() {
  vector<Particle> resampled_particles;
  default_random_engine gen;
  discrete_distribution<int> index(weights.begin(), weights.end());

  for(unsigned int i = 0; i < particles.size(); ++i) {
    int j = index(gen);
    Particle p {
      j,
      particles[j].x,
      particles[j].y,
      particles[i].theta,
      1
    };

    resampled_particles.push_back(p);
  }

  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
