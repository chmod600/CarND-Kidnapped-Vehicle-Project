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

using namespace std;

void ParticleFilter::init(double gps_x, double gps_y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 50;
  particles.resize(num_particles);
  default_random_engine gen;
  double std_x, std_y, std_theta;

  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  normal_distribution<double> dist_x(gps_x, std_x);
  normal_distribution<double> dist_y(gps_y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for(int i = 0; i < num_particles; ++i) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
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

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {

  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  for(unsigned long i = 0; i < observations.size(); ++i) {

    double current_min = 999999.0;
    unsigned long min_index;

    for(unsigned long j = 0; j < predicted.size(): ++j) {
      double dist_x = pow((predicted[j].x - observations[i].x), 2.0);
      double dist_y = pow((predicted[j].y - observations[i].y), 2.0);
      double dist = sqrt(dist_x + dist_y);

      if(dist < current_min) {
        min_index = j;
        current_min = dist;
      }
    }

    observations[i].id = min_index;
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
  std::vector<LandmarkObs> landmarks_within_range;

  double w_deno = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

  for(unsigned int i = 0; i < particles.size(); ++i) {
    for(unsigned j = 0; j < observations.size(); ++j) {

      double trans_x = particles[i].x +
        (cos(particles[i].theta) * observations[j].x) -
        (sin(particles[i].theta) * observations[j].y);

      double trans_y = particles[i].y +
        (sin(particles[i].theta) * observations[j].x) -
        (cos(particles[i].theta) * observations[j].y);

      LandmarkObs trans_observation = {
        observations[i].id,
        trans_x,
        trans_y,
      };

      trans_observations.push_back(trans_observation);
    }

    // 2. Associate each observation with landmark (use dataAssociation function above)
    // Map Landmarks are ground truth
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      double lm_x = map_landmarks.landmark_list[j].x_f;
      double lm_y = map_landmarks.landmark_list[j].y_f;

      double dist_x = lm_x - particles[i].x;
      double dist_y = lm_y - particles[i].y;
      double dist = sqrt(pow(dist_x, 2) + pow(dist_y, 2));

      if (dist < sensor_range) {
        LandmarkObs landmark_within_range = {
          map_landmarks.landmark_list[j].id_i,
          lm_x,
          lm_y,
        };

        landmarks_within_range.push_back(landmark_within_range);
      }
    }

    dataAssociation(landmarks_within_range, trans_observations);

    double particle_weight = 1;

    for(unsigned int j = 0; j < trans_observations.size(); ++j) {
      double x = trans_observations[j].x - landmarks_within_range[j].x;
      double y = trans_observations[j].y - landmarks_within_range[j].y;
      double exponent1 = (pow(x, 2) / (2 * pow(std_landmark[0], 2)));
      double exponent2 = (pow(y, 2) / (2 * pow(std_landmark[1], 2)));
      double exponent = - (exponent1 + exponent2);
      double weight = w_deno * exp(-exponent);
      particle_weight *= weight;
    }

    particles[i].weight = particle_weight;
    weights[i] = particle_weight;
  } // For each particle
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
