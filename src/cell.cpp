
#include "openmc/cell.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <iterator>
#include <set>
#include <sstream>
#include <string>

#include <fmt/core.h>

#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/dagmc.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/hdf5_interface.h"
#include "openmc/lattice.h"
#include "openmc/material.h"
#include "openmc/model/geometry/compiled_geometry.h"
#include "openmc/nuclide.h"
#include "openmc/settings.h"
#include "openmc/xml_interface.h"

namespace openmc {

namespace {

BoundingBox conservative_halfspace_bound(const Surface& surface, bool positive)
{
  BoundingBox bbox = surface.bounding_box(positive);
  double padding {0.0};
  if (dynamic_cast<const SurfaceXPlane*>(&surface) ||
      dynamic_cast<const SurfaceYPlane*>(&surface) ||
      dynamic_cast<const SurfaceZPlane*>(&surface)) {
    padding = FP_COINCIDENT;
  } else if (!positive) {
    double radius {-1.0};
    if (const auto* cylinder = dynamic_cast<const SurfaceXCylinder*>(&surface)) {
      radius = cylinder->radius_;
    } else if (const auto* cylinder =
                 dynamic_cast<const SurfaceYCylinder*>(&surface)) {
      radius = cylinder->radius_;
    } else if (const auto* cylinder =
                 dynamic_cast<const SurfaceZCylinder*>(&surface)) {
      radius = cylinder->radius_;
    } else if (const auto* sphere =
                 dynamic_cast<const SurfaceSphere*>(&surface)) {
      radius = sphere->radius_;
    }
    if (radius >= 0.0) {
      padding = FP_COINCIDENT /
                (std::sqrt(radius * radius + FP_COINCIDENT) + radius);
    } else {
      // An unknown finite bound cannot account for Surface::sense tolerance.
      return BoundingBox::infinite();
    }
  }

  for (int axis = 0; axis < 3; ++axis) {
    if (bbox.min[axis] > -INFTY) {
      bbox.min[axis] =
        std::nextafter(bbox.min[axis] - padding, -INFINITY);
    }
    if (bbox.max[axis] < INFTY) {
      bbox.max[axis] = std::nextafter(bbox.max[axis] + padding, INFINITY);
    }
  }
  return bbox;
}

BoundingBox conservative_surface_bound(const Surface& surface)
{
  BoundingBox bbox =
    surface.bounding_box(false) & surface.bounding_box(true);
  for (int axis = 0; axis < 3; ++axis) {
    if (bbox.min[axis] > -INFTY) {
      bbox.min[axis] =
        std::nextafter(bbox.min[axis] - FP_COINCIDENT, -INFINITY);
    }
    if (bbox.max[axis] < INFTY) {
      bbox.max[axis] =
        std::nextafter(bbox.max[axis] + FP_COINCIDENT, INFINITY);
    }
  }
  return bbox;
}

} // namespace

//==============================================================================
// Global variables
//==============================================================================

namespace model {
std::unordered_map<int32_t, int32_t> cell_map;
vector<unique_ptr<Cell>> cells;

} // namespace model

//==============================================================================
// Cell implementation
//==============================================================================

int32_t Cell::n_instances() const
{
  return model::universes[universe_]->n_instances_;
}

void Cell::set_rotation(const vector<double>& rot)
{
  if (fill_ == C_NONE) {
    fatal_error(fmt::format("Cannot apply a rotation to cell {}"
                            " because it is not filled with another universe",
      id_));
  }

  if (rot.size() != 3 && rot.size() != 9) {
    fatal_error(fmt::format("Non-3D rotation vector applied to cell {}", id_));
  }

  // Compute and store the inverse rotation matrix for the angles given.
  rotation_.clear();
  rotation_.reserve(rot.size() == 9 ? 9 : 12);
  if (rot.size() == 3) {
    double phi = -rot[0] * PI / 180.0;
    double theta = -rot[1] * PI / 180.0;
    double psi = -rot[2] * PI / 180.0;
    rotation_.push_back(std::cos(theta) * std::cos(psi));
    rotation_.push_back(-std::cos(phi) * std::sin(psi) +
                        std::sin(phi) * std::sin(theta) * std::cos(psi));
    rotation_.push_back(std::sin(phi) * std::sin(psi) +
                        std::cos(phi) * std::sin(theta) * std::cos(psi));
    rotation_.push_back(std::cos(theta) * std::sin(psi));
    rotation_.push_back(std::cos(phi) * std::cos(psi) +
                        std::sin(phi) * std::sin(theta) * std::sin(psi));
    rotation_.push_back(-std::sin(phi) * std::cos(psi) +
                        std::cos(phi) * std::sin(theta) * std::sin(psi));
    rotation_.push_back(-std::sin(theta));
    rotation_.push_back(std::sin(phi) * std::cos(theta));
    rotation_.push_back(std::cos(phi) * std::cos(theta));

    // When user specifies angles, write them at end of vector
    rotation_.push_back(rot[0]);
    rotation_.push_back(rot[1]);
    rotation_.push_back(rot[2]);
  } else {
    std::copy(rot.begin(), rot.end(), std::back_inserter(rotation_));
  }
}

double Cell::temperature(int32_t instance) const
{
  if (sqrtkT_.size() < 1) {
    throw std::runtime_error {"Cell temperature has not yet been set."};
  }

  if (instance >= 0) {
    double sqrtkT = sqrtkT_.size() == 1 ? sqrtkT_.at(0) : sqrtkT_.at(instance);
    return sqrtkT * sqrtkT / K_BOLTZMANN;
  } else {
    return sqrtkT_[0] * sqrtkT_[0] / K_BOLTZMANN;
  }
}

double Cell::density_mult(int32_t instance) const
{
  if (instance >= 0) {
    return density_mult_.size() == 1 ? density_mult_.at(0)
                                     : density_mult_.at(instance);
  } else {
    return density_mult_[0];
  }
}

double Cell::density(int32_t instance) const
{
  const int32_t mat_index = material(instance);
  if (mat_index == MATERIAL_VOID)
    return 0.0;

  return density_mult(instance) * model::materials[mat_index]->density_gpcc();
}

void Cell::set_temperature(double T, int32_t instance, bool set_contained)
{
  if (settings::temperature_method == TemperatureMethod::INTERPOLATION) {
    if (T < (data::temperature_min - settings::temperature_tolerance)) {
      throw std::runtime_error {
        fmt::format("Temperature of {} K is below minimum temperature at "
                    "which data is available of {} K.",
          T, data::temperature_min)};
    } else if (T > (data::temperature_max + settings::temperature_tolerance)) {
      throw std::runtime_error {
        fmt::format("Temperature of {} K is above maximum temperature at "
                    "which data is available of {} K.",
          T, data::temperature_max)};
    }
  }

  if (type_ == Fill::MATERIAL) {
    if (instance >= 0) {
      // If temperature vector is not big enough, resize it first
      if (sqrtkT_.size() != n_instances())
        sqrtkT_.resize(n_instances(), sqrtkT_[0]);

      // Set temperature for the corresponding instance
      sqrtkT_.at(instance) = std::sqrt(K_BOLTZMANN * T);
    } else {
      // Set temperature for all instances
      for (auto& T_ : sqrtkT_) {
        T_ = std::sqrt(K_BOLTZMANN * T);
      }
    }
  } else {
    if (!set_contained) {
      throw std::runtime_error {
        fmt::format("Attempted to set the temperature of cell {} "
                    "which is not filled by a material.",
          id_)};
    }

    auto contained_cells = this->get_contained_cells(instance);
    for (const auto& entry : contained_cells) {
      auto& cell = model::cells[entry.first];
      assert(cell->type_ == Fill::MATERIAL);
      auto& instances = entry.second;
      for (auto instance : instances) {
        cell->set_temperature(T, instance);
      }
    }
  }
}

void Cell::set_density(double density, int32_t instance, bool set_contained)
{
  if (type_ != Fill::MATERIAL && !set_contained) {
    fatal_error(
      fmt::format("Attempted to set the density multiplier of cell {} "
                  "which is not filled by a material.",
        id_));
  }

  if (type_ == Fill::MATERIAL) {
    const int32_t mat_index = material(instance);
    if (mat_index == MATERIAL_VOID)
      return;

    if (instance >= 0) {
      // If density multiplier vector is not big enough, resize it first
      if (density_mult_.size() != n_instances())
        density_mult_.resize(n_instances(), density_mult_[0]);

      // Set density multiplier for the corresponding instance
      density_mult_.at(instance) =
        density / model::materials[mat_index]->density_gpcc();
    } else {
      // Set density multiplier for all instances
      for (auto& x : density_mult_) {
        x = density / model::materials[mat_index]->density_gpcc();
      }
    }
  } else {
    auto contained_cells = this->get_contained_cells(instance);
    for (const auto& entry : contained_cells) {
      auto& cell = model::cells[entry.first];
      assert(cell->type_ == Fill::MATERIAL);
      auto& instances = entry.second;
      for (auto instance : instances) {
        cell->set_density(density, instance);
      }
    }
  }
}

void Cell::export_properties_hdf5(hid_t group) const
{
  // Create a group for this cell.
  auto cell_group = create_group(group, fmt::format("cell {}", id_));

  // Write temperature in [K] for one or more cell instances
  vector<double> temps;
  for (auto sqrtkT_val : sqrtkT_)
    temps.push_back(sqrtkT_val * sqrtkT_val / K_BOLTZMANN);
  write_dataset(cell_group, "temperature", temps);

  // Write density for one or more cell instances
  if (type_ == Fill::MATERIAL && material_.size() > 0) {
    vector<double> density;
    for (int32_t i = 0; i < density_mult_.size(); ++i)
      density.push_back(this->density(i));

    write_dataset(cell_group, "density", density);
  }

  close_group(cell_group);
}

void Cell::import_properties_hdf5(hid_t group)
{
  auto cell_group = open_group(group, fmt::format("cell {}", id_));

  // Read temperatures from file
  vector<double> temps;
  read_dataset(cell_group, "temperature", temps);

  // Ensure number of temperatures makes sense
  auto n_temps = temps.size();
  if (n_temps > 1 && n_temps != n_instances()) {
    fatal_error(fmt::format(
      "Number of temperatures for cell {} doesn't match number of instances",
      id_));
  }

  // Modify temperatures for the cell
  sqrtkT_.clear();
  sqrtkT_.resize(temps.size());
  for (int64_t i = 0; i < temps.size(); ++i) {
    this->set_temperature(temps[i], i);
  }

  // Read densities
  if (object_exists(cell_group, "density")) {
    vector<double> density;
    read_dataset(cell_group, "density", density);

    // Ensure number of densities makes sense
    auto n_density = density.size();
    if (n_density > 1 && n_density != n_instances()) {
      fatal_error(fmt::format("Number of densities for cell {} "
                              "doesn't match number of instances",
        id_));
    }

    // Set densities.
    for (int32_t i = 0; i < n_density; ++i) {
      this->set_density(density[i], i);
    }
  }

  close_group(cell_group);
}

void Cell::to_hdf5(hid_t cell_group) const
{

  // Create a group for this cell.
  auto group = create_group(cell_group, fmt::format("cell {}", id_));

  if (!name_.empty()) {
    write_string(group, "name", name_, false);
  }

  write_dataset(group, "universe", model::universes[universe_]->id_);

  to_hdf5_inner(group);

  // Write fill information.
  if (type_ == Fill::MATERIAL) {
    write_dataset(group, "fill_type", "material");
    std::vector<int32_t> mat_ids;
    for (auto i_mat : material_) {
      if (i_mat != MATERIAL_VOID) {
        mat_ids.push_back(model::materials[i_mat]->id_);
      } else {
        mat_ids.push_back(MATERIAL_VOID);
      }
    }
    if (mat_ids.size() == 1) {
      write_dataset(group, "material", mat_ids[0]);
    } else {
      write_dataset(group, "material", mat_ids);
    }

    std::vector<double> temps;
    for (auto sqrtkT_val : sqrtkT_)
      temps.push_back(sqrtkT_val * sqrtkT_val / K_BOLTZMANN);
    write_dataset(group, "temperature", temps);

    write_dataset(group, "density_mult", density_mult_);

  } else if (type_ == Fill::UNIVERSE) {
    write_dataset(group, "fill_type", "universe");
    write_dataset(group, "fill", model::universes[fill_]->id_);
    if (translation_ != Position(0, 0, 0)) {
      write_dataset(group, "translation", translation_);
    }
    if (!rotation_.empty()) {
      if (rotation_.size() == 12) {
        std::array<double, 3> rot {rotation_[9], rotation_[10], rotation_[11]};
        write_dataset(group, "rotation", rot);
      } else {
        write_dataset(group, "rotation", rotation_);
      }
    }

  } else if (type_ == Fill::LATTICE) {
    write_dataset(group, "fill_type", "lattice");
    write_dataset(group, "lattice", model::lattices[fill_]->id_);
  }

  close_group(group);
}

//==============================================================================
// XML parsing helpers for <cell> nodes
//==============================================================================

vector<int32_t> parse_cell_material_xml(pugi::xml_node node, int32_t cell_id)
{
  vector<std::string> mats {
    get_node_array<std::string>(node, "material", true)};
  if (mats.empty()) {
    fatal_error(fmt::format(
      "An empty material element was specified for cell {}", cell_id));
  }
  vector<int32_t> material;
  material.reserve(mats.size());
  for (const auto& mat : mats) {
    if (mat == "void") {
      material.push_back(MATERIAL_VOID);
    } else {
      material.push_back(std::stoi(mat));
    }
  }
  return material;
}

vector<double> parse_cell_temperature_xml(pugi::xml_node node, int32_t cell_id)
{
  auto temperatures = get_node_array<double>(node, "temperature");
  if (temperatures.empty()) {
    fatal_error(fmt::format(
      "An empty temperature element was specified for cell {}", cell_id));
  }
  for (auto T : temperatures) {
    if (T < 0) {
      fatal_error(fmt::format(
        "Cell {} was specified with a negative temperature", cell_id));
    }
  }
  return temperatures;
}

vector<double> parse_cell_density_xml(pugi::xml_node node, int32_t cell_id)
{
  auto densities = get_node_array<double>(node, "density");
  if (densities.empty()) {
    fatal_error(fmt::format(
      "An empty density element was specified for cell {}", cell_id));
  }
  for (auto rho : densities) {
    if (rho <= 0) {
      fatal_error(fmt::format(
        "Cell {} was specified with a density less than or equal to zero",
        cell_id));
    }
  }
  return densities;
}

//==============================================================================
// CSGCell implementation
//==============================================================================

CSGCell::CSGCell(pugi::xml_node cell_node)
{
  if (check_for_node(cell_node, "id")) {
    id_ = std::stoi(get_node_value(cell_node, "id"));
  } else {
    fatal_error("Must specify id of cell in geometry XML file.");
  }

  if (check_for_node(cell_node, "name")) {
    name_ = get_node_value(cell_node, "name");
  }

  if (check_for_node(cell_node, "universe")) {
    universe_ = std::stoi(get_node_value(cell_node, "universe"));
  } else {
    universe_ = 0;
  }

  // Make sure that either material or fill was specified, but not both.
  bool fill_present = check_for_node(cell_node, "fill");
  bool material_present = check_for_node(cell_node, "material");
  if (!(fill_present || material_present)) {
    fatal_error(
      fmt::format("Neither material nor fill was specified for cell {}", id_));
  }
  if (fill_present && material_present) {
    fatal_error(fmt::format("Cell {} has both a material and a fill specified; "
                            "only one can be specified per cell",
      id_));
  }

  if (fill_present) {
    fill_ = std::stoi(get_node_value(cell_node, "fill"));
    if (fill_ == universe_) {
      fatal_error(fmt::format("Cell {} is filled with the same universe that "
                              "it is contained in.",
        id_));
    }
  } else {
    fill_ = C_NONE;
  }

  // Read the material element.  There can be zero materials (filled with a
  // universe), more than one material (distribmats), and some materials may
  // be "void".
  if (material_present) {
    material_ = parse_cell_material_xml(cell_node, id_);
  }

  // Read the temperature element which may be distributed like materials.
  if (check_for_node(cell_node, "temperature")) {
    sqrtkT_ = parse_cell_temperature_xml(cell_node, id_);
    sqrtkT_.shrink_to_fit();

    // Make sure this is a material-filled cell.
    if (material_.size() == 0) {
      fatal_error(fmt::format(
        "Cell {} was specified with a temperature but no material. Temperature"
        "specification is only valid for cells filled with a material.",
        id_));
    }

    // Convert to sqrt(k*T).
    for (auto& T : sqrtkT_) {
      T = std::sqrt(K_BOLTZMANN * T);
    }
  }

  // Read the density element which can be distributed similar to temperature.
  // These get assigned to the density multiplier, requiring a division by
  // the material density.
  // Note: calculating the actual density multiplier is deferred until materials
  // are finalized. density_mult_ contains the true density in the meantime.
  if (check_for_node(cell_node, "density")) {
    density_mult_ = parse_cell_density_xml(cell_node, id_);
    density_mult_.shrink_to_fit();

    // Make sure this is a material-filled cell.
    if (material_.size() == 0) {
      fatal_error(fmt::format(
        "Cell {} was specified with a density but no material. Density"
        "specification is only valid for cells filled with a material.",
        id_));
    }

    // Make sure this is a non-void material.
    for (auto mat_id : material_) {
      if (mat_id == MATERIAL_VOID) {
        fatal_error(fmt::format(
          "Cell {} was specified with a density, but contains a void "
          "material. Density specification is only valid for cells "
          "filled with a non-void material.",
          id_));
      }
    }
  }

  // Read the region specification.
  std::string region_spec;
  if (check_for_node(cell_node, "region")) {
    region_spec = get_node_value(cell_node, "region");
  }

  // Get a tokenized representation of the region specification and apply De
  // Morgans law
  Region region(region_spec, id_);
  region_ = region;

  // Read the translation vector.
  if (check_for_node(cell_node, "translation")) {
    if (fill_ == C_NONE) {
      fatal_error(fmt::format("Cannot apply a translation to cell {}"
                              " because it is not filled with another universe",
        id_));
    }

    auto xyz {get_node_array<double>(cell_node, "translation")};
    if (xyz.size() != 3) {
      fatal_error(
        fmt::format("Non-3D translation vector applied to cell {}", id_));
    }
    translation_ = xyz;
  }

  // Read the rotation transform.
  if (check_for_node(cell_node, "rotation")) {
    auto rot {get_node_array<double>(cell_node, "rotation")};
    set_rotation(rot);
  }
}

//==============================================================================

void CSGCell::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "geom_type", "csg", false);
  write_string(group_id, "region", region_.str(), false);
}

//==============================================================================

vector<int32_t>::iterator CSGCell::find_left_parenthesis(
  vector<int32_t>::iterator start, const vector<int32_t>& infix)
{
  // start search at zero
  int parenthesis_level = 0;
  auto it = start;
  while (it != infix.begin()) {
    // look at two tokens at a time
    int32_t one = *it;
    int32_t two = *(it - 1);

    // decrement parenthesis level if there are two adjacent surfaces
    if (one < OP_UNION && two < OP_UNION) {
      parenthesis_level--;
      // increment if there are two adjacent operators
    } else if (one >= OP_UNION && two >= OP_UNION) {
      parenthesis_level++;
    }

    // if the level gets to zero, return the position
    if (parenthesis_level == 0) {
      // move the iterator back one before leaving the loop
      // so that all tokens in the parenthesis block are included
      it--;
      break;
    }

    // continue loop, one token at a time
    it--;
  }
  return it;
}

//==============================================================================
// Region implementation
//==============================================================================

Region::Region(std::string region_spec, int32_t cell_id)
{
  // Check if region_spec is not empty.
  if (!region_spec.empty()) {
    // Parse all halfspaces and operators except for intersection (whitespace).
    for (int i = 0; i < region_spec.size();) {
      if (region_spec[i] == '(') {
        expression_.push_back(OP_LEFT_PAREN);
        i++;

      } else if (region_spec[i] == ')') {
        expression_.push_back(OP_RIGHT_PAREN);
        i++;

      } else if (region_spec[i] == '|') {
        expression_.push_back(OP_UNION);
        i++;

      } else if (region_spec[i] == '~') {
        expression_.push_back(OP_COMPLEMENT);
        i++;

      } else if (region_spec[i] == '-' || region_spec[i] == '+' ||
                 std::isdigit(region_spec[i])) {
        // This is the start of a halfspace specification.  Iterate j until we
        // find the end, then push-back everything between i and j.
        int j = i + 1;
        while (j < region_spec.size() && std::isdigit(region_spec[j])) {
          j++;
        }
        expression_.push_back(std::stoi(region_spec.substr(i, j - i)));
        i = j;

      } else if (std::isspace(region_spec[i])) {
        i++;

      } else {
        auto err_msg =
          fmt::format("Region specification contains invalid character, \"{}\"",
            region_spec[i]);
        fatal_error(err_msg);
      }
    }

    // Add in intersection operators where a missing operator is needed.
    int i = 0;
    while (i < expression_.size() - 1) {
      bool left_compat {
        (expression_[i] < OP_UNION) || (expression_[i] == OP_RIGHT_PAREN)};
      bool right_compat {(expression_[i + 1] < OP_UNION) ||
                         (expression_[i + 1] == OP_LEFT_PAREN) ||
                         (expression_[i + 1] == OP_COMPLEMENT)};
      if (left_compat && right_compat) {
        expression_.insert(expression_.begin() + i + 1, OP_INTERSECTION);
      }
      i++;
    }

    // Remove complement operators using DeMorgan's laws
    auto it = std::find(expression_.begin(), expression_.end(), OP_COMPLEMENT);
    while (it != expression_.end()) {
      // Erase complement. Note that erase invalidates the iterator, so we have
      // to use the iterator it returns, which points to the token that
      // followed the complement operator.
      it = expression_.erase(it);
      if (it == expression_.end())
        break;

      // Define stop given left parenthesis or not
      auto stop = it;
      if (*it == OP_LEFT_PAREN) {
        int depth = 1;
        do {
          stop++;
          if (*stop > OP_COMPLEMENT) {
            if (*stop == OP_RIGHT_PAREN) {
              depth--;
            } else {
              depth++;
            }
          }
        } while (depth > 0);
        it++;
      }

      // apply DeMorgan's law to any surfaces/operators between these
      // positions in the RPN
      apply_demorgan(it, stop);
      // update iterator position
      it = std::find(expression_.begin(), expression_.end(), OP_COMPLEMENT);
    }

    // Convert user IDs to surface indices.
    for (auto& r : expression_) {
      if (r < OP_UNION) {
        const auto& it {model::surface_map.find(abs(r))};
        if (it == model::surface_map.end()) {
          throw std::runtime_error {
            "Invalid surface ID " + std::to_string(abs(r)) +
            " specified in region for cell " + std::to_string(cell_id) + "."};
        }
        r = (r > 0) ? it->second + 1 : -(it->second + 1);
      }
    }

    // Check if this is a simple cell.
    simple_ = true;
    for (int32_t token : expression_) {
      if (token == OP_UNION) {
        simple_ = false;
        // Ensure intersections have precedence over unions
        enforce_precedence();
        break;
      }
    }

    // If this cell is simple, remove all the superfluous operator tokens.
    if (simple_) {
      expression_.erase(std::remove_if(expression_.begin(), expression_.end(),
                          [](int32_t token) {
                            return token == OP_INTERSECTION ||
                                   token > OP_COMPLEMENT;
                          }),
        expression_.end());
    }
    expression_.shrink_to_fit();

    // Distance calculations only depend on the geometric surface, not how many
    // times or with which sense it appears in the Boolean expression.
    std::unordered_set<int32_t> seen_surfaces;
    for (int32_t token : expression_) {
      if (token < OP_UNION && seen_surfaces.insert(std::abs(token)).second) {
        surface_tokens_.push_back(token);
        const Surface* surface = model::surfaces[std::abs(token) - 1].get();
        surface_bounds_.push_back(conservative_surface_bound(*surface));
        PlaneKernel kernel;
        if (const auto* plane = dynamic_cast<const SurfaceXPlane*>(surface)) {
          kernel.type = PlaneType::X;
          kernel.coefficients[0] = plane->x0_;
        } else if (const auto* plane =
                    dynamic_cast<const SurfaceYPlane*>(surface)) {
          kernel.type = PlaneType::Y;
          kernel.coefficients[0] = plane->y0_;
        } else if (const auto* plane =
                    dynamic_cast<const SurfaceZPlane*>(surface)) {
          kernel.type = PlaneType::Z;
          kernel.coefficients[0] = plane->z0_;
        } else if (const auto* plane =
                    dynamic_cast<const SurfacePlane*>(surface)) {
          kernel.type = PlaneType::GENERAL;
          kernel.coefficients[0] = plane->A_;
          kernel.coefficients[1] = plane->B_;
          kernel.coefficients[2] = plane->C_;
          kernel.coefficients[3] = plane->D_;
        } else {
          all_surfaces_are_planes_ = false;
        }
        plane_kernels_.push_back(kernel);
      }
    }
    surface_tokens_.shrink_to_fit();
    if (simple_) {
      plane_kernels_.clear();
    }
    plane_kernels_.shrink_to_fit();

    if (!simple_) {
      std::unordered_map<int32_t, std::size_t> surface_positions;
      surface_leaf_nodes_.resize(surface_tokens_.size());
      for (std::size_t i = 0; i < surface_tokens_.size(); ++i) {
        surface_positions[std::abs(surface_tokens_[i])] = i;
      }

      vector<int> node_stack;
      for (int32_t token : generate_postfix(cell_id)) {
        if (token < OP_UNION) {
          const int node = boolean_nodes_.size();
          const std::size_t surface = surface_positions.at(std::abs(token));
          const auto& surf = *model::surfaces[std::abs(token) - 1];
          const BoundingBox bbox =
            conservative_halfspace_bound(surf, token > 0);
          boolean_nodes_.push_back({token, -1, -1, -1, surface, bbox});
          surface_leaf_nodes_[surface].push_back(node);
          node_stack.push_back(node);
        } else {
          const int right = node_stack.back();
          node_stack.pop_back();
          const int left = node_stack.back();
          node_stack.pop_back();
          const int node = boolean_nodes_.size();
          const BoundingBox bbox = token == OP_INTERSECTION
                                     ? boolean_nodes_[left].bbox &
                                         boolean_nodes_[right].bbox
                                     : boolean_nodes_[left].bbox |
                                         boolean_nodes_[right].bbox;
          boolean_nodes_.push_back({token, left, right, -1, 0, bbox});
          boolean_nodes_[left].parent = node;
          boolean_nodes_[right].parent = node;
          node_stack.push_back(node);
        }
      }
      boolean_root_ = node_stack.back();
      balance_boolean_tree();
    }

    short_circuit_jump_.assign(expression_.size(), expression_.size());
    vector<vector<std::size_t>> operators_by_depth;
    for (std::size_t i = 0; i < expression_.size(); ++i) {
      if (expression_[i] == OP_LEFT_PAREN) {
        operators_by_depth.emplace_back();
      } else if ((expression_[i] == OP_UNION ||
                   expression_[i] == OP_INTERSECTION) &&
                 !operators_by_depth.empty()) {
        operators_by_depth.back().push_back(i);
      } else if (expression_[i] == OP_RIGHT_PAREN &&
                 !operators_by_depth.empty()) {
        for (std::size_t op : operators_by_depth.back()) {
          short_circuit_jump_[op] = i;
        }
        operators_by_depth.pop_back();
      }
    }

  } else {
    simple_ = true;
  }
}

//==============================================================================

void Region::balance_boolean_tree()
{
  vector<BooleanNode> old_nodes = std::move(boolean_nodes_);
  boolean_nodes_.clear();
  boolean_nodes_.reserve(old_nodes.size());
  for (auto& leaves : surface_leaf_nodes_)
    leaves.clear();

  std::function<void(int, int32_t, vector<int>&)> collect_operands;
  collect_operands = [&](int node, int32_t token, vector<int>& operands) {
    const auto& item = old_nodes[node];
    if (item.token == token) {
      collect_operands(item.left, token, operands);
      collect_operands(item.right, token, operands);
    } else {
      operands.push_back(node);
    }
  };

  std::function<int(int)> rebuild;
  std::function<int(const vector<int>&, std::size_t, std::size_t, int32_t)>
    join;
  join = [&](const vector<int>& operands, std::size_t begin, std::size_t end,
           int32_t token) {
    if (end - begin == 1)
      return rebuild(operands[begin]);
    const std::size_t middle = begin + (end - begin) / 2;
    const int left = join(operands, begin, middle, token);
    const int right = join(operands, middle, end, token);
    const int node = boolean_nodes_.size();
    const BoundingBox bbox = token == OP_INTERSECTION
                               ? boolean_nodes_[left].bbox &
                                   boolean_nodes_[right].bbox
                               : boolean_nodes_[left].bbox |
                                   boolean_nodes_[right].bbox;
    boolean_nodes_.push_back({token, left, right, -1, 0, bbox});
    boolean_nodes_[left].parent = node;
    boolean_nodes_[right].parent = node;
    return node;
  };
  rebuild = [&](int old_node) {
    const auto& item = old_nodes[old_node];
    if (item.token < OP_UNION) {
      const int node = boolean_nodes_.size();
      boolean_nodes_.push_back(
        {item.token, -1, -1, -1, item.surface, item.bbox});
      surface_leaf_nodes_[item.surface].push_back(node);
      return node;
    }

    vector<int> operands;
    collect_operands(old_node, item.token, operands);
    return join(operands, 0, operands.size(), item.token);
  };
  boolean_root_ = rebuild(boolean_root_);
}

//==============================================================================

void Region::apply_demorgan(
  vector<int32_t>::iterator start, vector<int32_t>::iterator stop)
{
  do {
    if (*start < OP_UNION) {
      *start *= -1;
    } else if (*start == OP_UNION) {
      *start = OP_INTERSECTION;
    } else if (*start == OP_INTERSECTION) {
      *start = OP_UNION;
    }
    start++;
  } while (start < stop);
}

//==============================================================================
//! Add precedence for infix regions so intersections have higher
//! precedence than unions using parentheses.
//==============================================================================

void Region::add_parentheses(int64_t start)
{
  int32_t start_token = expression_[start];
  // Add left parenthesis and set new position to be after parenthesis
  if (start_token == OP_UNION) {
    start += 2;
  }
  expression_.insert(expression_.begin() + start - 1, OP_LEFT_PAREN);

  // Add right parenthesis
  // While the start iterator is within the bounds of infix
  while (start + 1 < expression_.size()) {
    start++;

    // If the current token is an operator and is different than the start token
    if (expression_[start] >= OP_UNION && expression_[start] != start_token) {
      // Skip wrapped regions but save iterator position to check precedence and
      // add right parenthesis, right parenthesis position depends on the
      // operator, when the operator is a union then do not include the operator
      // in the region, when the operator is an intersection then include the
      // operator and next surface
      if (expression_[start] == OP_LEFT_PAREN) {
        int depth = 1;
        do {
          start++;
          if (expression_[start] > OP_COMPLEMENT) {
            if (expression_[start] == OP_RIGHT_PAREN) {
              depth--;
            } else {
              depth++;
            }
          }
        } while (depth > 0);
      } else {
        if (start_token == OP_UNION) {
          --start;
        }
        expression_.insert(expression_.begin() + start, OP_RIGHT_PAREN);
        return;
      }
    }
  }
  // If we get here a right parenthesis hasn't been placed
  expression_.push_back(OP_RIGHT_PAREN);
}

//==============================================================================
//! Add parentheses to enforce operator precedence in region expressions
//!
//! This function ensures that intersection operators have higher precedence
//! than union operators by adding parentheses where needed. For example:
//!   "1 2 | 3" becomes "(1 2) | 3"
//!   "1 | 2 3" becomes "1 | (2 3)"
//!
//! The algorithm uses stacks to track the current operator type and its
//! position at each parenthesis depth level. When it encounters a different
//! operator at the same depth, it adds parentheses to group the
//! higher-precedence operations.
//==============================================================================

void Region::enforce_precedence()
{
  // Stack tracking the operator type at each depth (0 = no operator seen yet)
  vector<int32_t> op_stack = {0};

  // Stack tracking where the operator sequence started at each depth
  vector<std::size_t> pos_stack = {0};

  for (int64_t i = 0; i < expression_.size(); ++i) {
    int32_t token = expression_[i];

    if (token == OP_LEFT_PAREN) {
      // Entering a new parenthesis level - push new tracking state
      op_stack.push_back(0);
      pos_stack.push_back(0);
      continue;
    } else if (token == OP_RIGHT_PAREN) {
      // Exiting a parenthesis level - pop tracking state (keep at least one)
      if (op_stack.size() > 1) {
        op_stack.pop_back();
        pos_stack.pop_back();
      }
      continue;
    }

    if (token == OP_UNION || token == OP_INTERSECTION) {
      if (op_stack.back() == 0) {
        // First operator at this depth - record it and its position
        op_stack.back() = token;
        pos_stack.back() = i;
      } else if (token != op_stack.back()) {
        // Encountered a different operator at the same depth - need to add
        // parentheses to enforce precedence. Intersection has higher
        // precedence, so we parenthesize the intersection terms.
        if (op_stack.back() == OP_INTERSECTION) {
          add_parentheses(pos_stack.back());
        } else {
          add_parentheses(i);
        }

        // Restart the scan since we modified the expression
        i = -1; // Will be incremented to 0 by the for loop
        op_stack = {0};
        pos_stack = {0};
      }
    }
  }
}

//==============================================================================
//! Convert infix region specification to Reverse Polish Notation (RPN)
//!
//! This function uses the shunting-yard algorithm.
//==============================================================================

vector<int32_t> Region::generate_postfix(int32_t cell_id) const
{
  vector<int32_t> rpn;
  vector<int32_t> stack;

  for (int32_t token : expression_) {
    if (token < OP_UNION) {
      // If token is not an operator, add it to output
      rpn.push_back(token);
    } else if (token < OP_RIGHT_PAREN) {
      // Regular operators union, intersection, complement
      while (stack.size() > 0) {
        int32_t op = stack.back();

        if (op < OP_RIGHT_PAREN && ((token == OP_COMPLEMENT && token < op) ||
                                     (token != OP_COMPLEMENT && token <= op))) {
          // While there is an operator, op, on top of the stack, if the token
          // is left-associative and its precedence is less than or equal to
          // that of op or if the token is right-associative and its precedence
          // is less than that of op, move op to the output queue and push the
          // token on to the stack. Note that only complement is
          // right-associative.
          rpn.push_back(op);
          stack.pop_back();
        } else {
          break;
        }
      }

      stack.push_back(token);

    } else if (token == OP_LEFT_PAREN) {
      // If the token is a left parenthesis, push it onto the stack
      stack.push_back(token);

    } else {
      // If the token is a right parenthesis, move operators from the stack to
      // the output queue until reaching the left parenthesis.
      for (auto it = stack.rbegin(); *it != OP_LEFT_PAREN; it++) {
        // If we run out of operators without finding a left parenthesis, it
        // means there are mismatched parentheses.
        if (it == stack.rend()) {
          fatal_error(fmt::format(
            "Mismatched parentheses in region specification for cell {}",
            cell_id));
        }
        rpn.push_back(stack.back());
        stack.pop_back();
      }

      // Pop the left parenthesis.
      stack.pop_back();
    }
  }

  while (stack.size() > 0) {
    int32_t op = stack.back();

    // If the operator is a parenthesis it is mismatched.
    if (op >= OP_RIGHT_PAREN) {
      fatal_error(fmt::format(
        "Mismatched parentheses in region specification for cell {}", cell_id));
    }

    rpn.push_back(stack.back());
    stack.pop_back();
  }

  return rpn;
}

//==============================================================================

std::string Region::str() const
{
  std::stringstream region_spec {};
  if (!expression_.empty()) {
    for (int32_t token : expression_) {
      if (token == OP_LEFT_PAREN) {
        region_spec << " (";
      } else if (token == OP_RIGHT_PAREN) {
        region_spec << " )";
      } else if (token == OP_COMPLEMENT) {
        region_spec << " ~";
      } else if (token == OP_INTERSECTION) {
      } else if (token == OP_UNION) {
        region_spec << " |";
      } else {
        // Note the off-by-one indexing
        auto surf_id = model::surfaces[abs(token) - 1]->id_;
        region_spec << " " << ((token > 0) ? surf_id : -surf_id);
      }
    }
  }
  return region_spec.str();
}

//==============================================================================

std::pair<double, int32_t> Region::distance(
  Position r, Direction u, int32_t on_surface, bool known_inside,
  double max_distance) const
{
  if (simple_) {
    return distance_to_nearest_surface(
      r, u, on_surface, false, max_distance);
  }

  const auto& candidates = candidate_surfaces(r, u, max_distance);
  if (candidates.empty()) {
    if (known_inside || on_surface != 0) {
      return distance_complex_fallback(
        r, u, on_surface, known_inside, max_distance);
    }
    return {INFTY, std::numeric_limits<int32_t>::max()};
  } else if (all_surfaces_are_planes_) {
    return distance_complex_planes(
      r, u, on_surface, known_inside, max_distance, candidates);
  } else {
    return distance_complex(
      r, u, on_surface, known_inside, max_distance, candidates);
  }
}

//==============================================================================

std::pair<double, int32_t> Region::distance_to_nearest_surface(Position r,
  Direction u, int32_t on_surface, bool ignore_coincident_surfaces,
  double max_distance, const vector<std::size_t>* candidate_surfaces) const
{
  double min_dist {INFTY};
  int32_t i_surf {std::numeric_limits<int32_t>::max()};

  const std::size_t n =
    candidate_surfaces ? candidate_surfaces->size() : surface_tokens_.size();
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t surface =
      candidate_surfaces ? (*candidate_surfaces)[i] : i;
    const int32_t token = surface_tokens_[surface];
    // Calculate the distance to this surface.
    // Note the off-by-one indexing
    bool coincident {std::abs(token) == std::abs(on_surface)};
    double d {model::surfaces[abs(token) - 1]->distance(r, u, coincident)};

    // Different surface definitions can represent the same geometric surface.
    // When the ray is already known to be on a surface, ignore intersections
    // with other surfaces at the same location to avoid repeatedly crossing
    // between them due to roundoff.
    if (ignore_coincident_surfaces && d < FP_COINCIDENT)
      continue;
    if (d >= max_distance)
      continue;

    // Check if this distance is the new minimum.
    if (d < min_dist) {
      if (min_dist - d >= FP_PRECISION * min_dist) {
        min_dist = d;
        i_surf = -token;
      }
    }
  }

  return {min_dist, i_surf};
}

//==============================================================================

std::pair<double, int32_t> Region::distance_complex(
  Position r, Direction u, int32_t on_surface, bool known_inside,
  double max_distance, const vector<std::size_t>& candidate_surfaces) const
{
  struct SurfaceEvent {
    double distance;
    std::size_t surface;
  };
  struct EventCompare {
    bool operator()(const SurfaceEvent& a, const SurfaceEvent& b) const
    {
      return a.distance > b.distance ||
             (a.distance == b.distance && a.surface > b.surface);
    }
  };

  static thread_local vector<SurfaceEvent> events;
  events.clear();
  events.reserve(candidate_surfaces.size());
  for (std::size_t i : candidate_surfaces) {
    const int32_t token = surface_tokens_[i];
    const bool coincident {std::abs(token) == std::abs(on_surface)};
    const double distance =
      plane_kernels_[i].type != PlaneType::NONE
        ? plane_distance(i, r, u, coincident)
        : model::surfaces[std::abs(token) - 1]->distance(r, u, coincident);
    if ((on_surface == 0 || distance >= FP_COINCIDENT) &&
        distance < max_distance) {
      events.push_back({distance, i});
    }
  }
  EventCompare compare;
  std::make_heap(events.begin(), events.end(), compare);

  static thread_local vector<int8_t> boolean_values;
  boolean_values.assign(boolean_nodes_.size(), -1);
  if (on_surface != 0) {
    for (std::size_t i = 0; i < surface_tokens_.size(); ++i) {
      if (std::abs(surface_tokens_[i]) == std::abs(on_surface)) {
        set_surface_boolean_value(i, on_surface > 0, boolean_values);
        break;
      }
    }
  }
  const bool in_region = known_inside ||
                         evaluate_boolean_node(
                           boolean_root_, r, u, boolean_values);

  while (!events.empty()) {
    std::pop_heap(events.begin(), events.end(), compare);
    const SurfaceEvent event = events.back();
    events.pop_back();

    // Near-simultaneous roots can depend on expression order and on_surface
    // tie-breaking. Preserve those cases with the exact rescan traversal.
    if (!events.empty()) {
      const double next = events.front().distance;
      const double tolerance = std::max(FP_COINCIDENT,
        FP_PRECISION * std::max(std::abs(event.distance), std::abs(next)));
      if (next - event.distance < tolerance) {
        return distance_complex_fallback(
          r, u, on_surface, known_inside, max_distance, &candidate_surfaces);
      }
    }

    const int32_t token = surface_tokens_[event.surface];
    int32_t i_surf = std::abs(token);
    const Position r_hit = r + event.distance * u;
    const auto& surf {*model::surfaces[i_surf - 1]};
    const bool is_plane = plane_kernels_[event.surface].type != PlaneType::NONE;
    const Direction normal = is_plane ? Direction {} : surf.normal(r_hit);
    const double normal_projection = is_plane
                                       ? plane_normal_projection(event.surface, u)
                                       : u.dot(normal);
    const double normal_norm = is_plane ? 1.0 : normal.norm();
    if (std::abs(normal_projection) <= FP_PRECISION * normal_norm) {
      return distance_complex_fallback(
        r, u, on_surface, known_inside, max_distance, &candidate_surfaces);
    }
    if (normal_projection < 0.0) {
      i_surf = -i_surf;
    }

    set_surface_boolean_value(event.surface, i_surf > 0, boolean_values);
    if (evaluate_boolean_node(boolean_root_, r, u, boolean_values) !=
        in_region) {
      return {event.distance, i_surf};
    }

    const double next = surf.distance(r_hit, u, true);
    if (next < FP_COINCIDENT) {
      return distance_complex_fallback(
        r, u, on_surface, known_inside, max_distance, &candidate_surfaces);
    }
    if (next < INFTY - event.distance &&
        next + event.distance < max_distance) {
      events.push_back({event.distance + next, event.surface});
      std::push_heap(events.begin(), events.end(), compare);
    }
  }

  return {INFTY, std::numeric_limits<int32_t>::max()};
}

//==============================================================================

std::pair<double, int32_t> Region::distance_complex_fallback(
  Position r, Direction u, int32_t on_surface, bool known_inside,
  double max_distance, const vector<std::size_t>* candidate_surfaces) const
{
  const bool in_region = known_inside || contains_complex(r, u, on_surface);
  auto search = [&](const vector<std::size_t>* candidates) {
    Position current_r {r};
    int32_t current_surface {on_surface};
    double total_distance {0.0};

    while (true) {
      auto [distance, i_surf] = distance_to_nearest_surface(current_r, u,
        current_surface, current_surface != 0, max_distance - total_distance,
        candidates);
      if (distance == INFTY) {
        return std::pair<double, int32_t> {
          INFTY, std::numeric_limits<int32_t>::max()};
      }

      // Use the normal at the hit rather than a surface evaluation because
      // accumulated roundoff may put a curved-surface hit on the wrong side.
      current_r += distance * u;
      total_distance += distance;
      i_surf = std::abs(i_surf);
      const auto& surf {*model::surfaces[i_surf - 1]};
      if (u.dot(surf.normal(current_r)) <= 0.0)
        i_surf = -i_surf;

      if (contains_complex(current_r, u, i_surf) != in_region)
        return std::pair<double, int32_t> {total_distance, i_surf};
      current_surface = i_surf;
    }
  };

  return search(candidate_surfaces);
}

//==============================================================================

bool Region::evaluate_boolean_node(
  int node, Position r, Direction u, vector<int8_t>& values) const
{
  if (values[node] >= 0) {
    return values[node];
  }

  const auto& item = boolean_nodes_[node];
  bool value;
  if (item.token < OP_UNION) {
    const bool sense = plane_kernels_[item.surface].type != PlaneType::NONE
                         ? plane_sense(item.surface, r, u)
                         : model::surfaces[std::abs(item.token) - 1]->sense(r, u);
    set_surface_boolean_value(item.surface, sense, values);
    value = values[node];
  } else if (item.token == OP_INTERSECTION) {
    value = evaluate_boolean_node(item.left, r, u, values) &&
            evaluate_boolean_node(item.right, r, u, values);
  } else {
    value = evaluate_boolean_node(item.left, r, u, values) ||
            evaluate_boolean_node(item.right, r, u, values);
  }
  values[node] = value;
  return value;
}

//==============================================================================

const vector<std::size_t>& Region::candidate_surfaces(
  Position r, Direction u, double max_distance) const
{
  struct WorkItem {
    int node;
    double entry;
    double exit;
  };
  struct Scratch {
    vector<std::size_t> candidates;
    vector<WorkItem> stack;
    vector<uint32_t> marks;
    uint32_t epoch {0};
  };
  static thread_local Scratch scratch;

  scratch.candidates.clear();
  scratch.stack.clear();
  scratch.candidates.reserve(surface_tokens_.size());
  scratch.stack.reserve(boolean_nodes_.size());
  if (scratch.marks.size() < surface_tokens_.size())
    scratch.marks.resize(surface_tokens_.size(), 0);
  if (++scratch.epoch == 0) {
    std::fill(scratch.marks.begin(), scratch.marks.end(), 0);
    ++scratch.epoch;
  }

  double entry {0.0};
  double exit {max_distance};
  if (!intersect_bound(boolean_nodes_[boolean_root_].bbox, r, u, entry, exit))
    return scratch.candidates;
  scratch.stack.push_back({boolean_root_, entry, exit});

  while (!scratch.stack.empty()) {
    const WorkItem work = scratch.stack.back();
    scratch.stack.pop_back();
    const auto& node = boolean_nodes_[work.node];
    if (node.token < OP_UNION) {
      double surface_entry {work.entry};
      double surface_exit {work.exit};
      if (intersect_bound(surface_bounds_[node.surface], r, u, surface_entry,
            surface_exit) &&
          scratch.marks[node.surface] != scratch.epoch) {
        scratch.marks[node.surface] = scratch.epoch;
        scratch.candidates.push_back(node.surface);
      }
    } else if (node.token == OP_INTERSECTION) {
      scratch.stack.push_back({node.right, work.entry, work.exit});
      scratch.stack.push_back({node.left, work.entry, work.exit});
    } else {
      double child_entry {work.entry};
      double child_exit {work.exit};
      if (intersect_bound(boolean_nodes_[node.right].bbox, r, u, child_entry,
            child_exit)) {
        scratch.stack.push_back({node.right, child_entry, child_exit});
      }
      child_entry = work.entry;
      child_exit = work.exit;
      if (intersect_bound(boolean_nodes_[node.left].bbox, r, u, child_entry,
            child_exit)) {
        scratch.stack.push_back({node.left, child_entry, child_exit});
      }
    }
  }
  return scratch.candidates;
}

//==============================================================================

bool Region::intersect_bound(const BoundingBox& bbox, Position r, Direction u,
  double& entry, double& exit) const
{
  if (bbox.min.x == -INFTY && bbox.min.y == -INFTY &&
      bbox.min.z == -INFTY && bbox.max.x == INFTY &&
      bbox.max.y == INFTY && bbox.max.z == INFTY)
    return true;

  // Check parallel slabs first. Disjoint union branches are commonly rejected
  // here without any divisions.
  for (int axis = 0; axis < 3; ++axis) {
    const double lower = bbox.min[axis];
    const double upper = bbox.max[axis];
    if (lower <= -INFTY && upper >= INFTY)
      continue;
    if (std::isnan(r[axis]) || std::isnan(u[axis]) || std::isnan(lower) ||
        std::isnan(upper))
      return true;
    if (lower > upper)
      return false;
    if (u[axis] == 0.0) {
      if (r[axis] < lower || r[axis] > upper)
        return false;
    }
  }

  for (int axis = 0; axis < 3; ++axis) {
    const double lower = bbox.min[axis];
    const double upper = bbox.max[axis];
    if (lower <= -INFTY && upper >= INFTY)
      continue;
    if (u[axis] == 0.0)
      continue;
    double axis_entry = (lower - r[axis]) / u[axis];
    double axis_exit = (upper - r[axis]) / u[axis];
    if (axis_entry > axis_exit)
      std::swap(axis_entry, axis_exit);
    entry = std::max(entry, axis_entry);
    exit = std::min(exit, axis_exit);
    if (entry > exit)
      return false;
  }
  return entry <= exit && exit >= 0.0;
}

//==============================================================================

void Region::set_surface_boolean_value(
  std::size_t surface, bool positive_side, vector<int8_t>& values) const
{
  for (int leaf : surface_leaf_nodes_[surface]) {
    values[leaf] = positive_side == (boolean_nodes_[leaf].token > 0);
    for (int parent = boolean_nodes_[leaf].parent; parent >= 0;
         parent = boolean_nodes_[parent].parent) {
      values[parent] = -1;
    }
  }
}

//==============================================================================

double Region::plane_distance(std::size_t surface, Position r, Direction u,
  bool coincident) const
{
  const auto& plane = plane_kernels_[surface];
  double f;
  double projection;
  switch (plane.type) {
  case PlaneType::X:
    f = plane.coefficients[0] - r.x;
    projection = u.x;
    break;
  case PlaneType::Y:
    f = plane.coefficients[0] - r.y;
    projection = u.y;
    break;
  case PlaneType::Z:
    f = plane.coefficients[0] - r.z;
    projection = u.z;
    break;
  case PlaneType::GENERAL:
    f = plane.coefficients[0] * r.x + plane.coefficients[1] * r.y +
        plane.coefficients[2] * r.z - plane.coefficients[3];
    projection = plane.coefficients[0] * u.x +
                 plane.coefficients[1] * u.y +
                 plane.coefficients[2] * u.z;
    if (coincident || std::abs(f) < FP_COINCIDENT || projection == 0.0)
      return INFTY;
    f = -f;
    break;
  default:
    return INFTY;
  }

  if (coincident || std::abs(f) < FP_COINCIDENT || projection == 0.0)
    return INFTY;
  const double distance = f / projection;
  return distance < 0.0 ? INFTY : distance;
}

//==============================================================================

double Region::plane_normal_projection(
  std::size_t surface, Direction u) const
{
  const auto& plane = plane_kernels_[surface];
  switch (plane.type) {
  case PlaneType::X:
    return u.x;
  case PlaneType::Y:
    return u.y;
  case PlaneType::Z:
    return u.z;
  case PlaneType::GENERAL:
    return plane.coefficients[0] * u.x + plane.coefficients[1] * u.y +
           plane.coefficients[2] * u.z;
  default:
    return 0.0;
  }
}

//==============================================================================

bool Region::plane_sense(
  std::size_t surface, Position r, Direction u) const
{
  const auto& plane = plane_kernels_[surface];
  double value;
  switch (plane.type) {
  case PlaneType::X:
    value = r.x - plane.coefficients[0];
    break;
  case PlaneType::Y:
    value = r.y - plane.coefficients[0];
    break;
  case PlaneType::Z:
    value = r.z - plane.coefficients[0];
    break;
  case PlaneType::GENERAL:
    value = plane.coefficients[0] * r.x + plane.coefficients[1] * r.y +
            plane.coefficients[2] * r.z - plane.coefficients[3];
    break;
  default:
    return false;
  }

  if (std::abs(value) < FP_COINCIDENT)
    return plane_normal_projection(surface, u) > 0.0;
  return value > 0.0;
}

//==============================================================================

std::pair<double, int32_t> Region::distance_complex_planes(Position r,
  Direction u, int32_t on_surface, bool known_inside, double max_distance,
  const vector<std::size_t>& candidate_surfaces) const
{
  struct Crossing {
    double distance;
    int32_t token;
    std::size_t surface;
    bool used;
  };

  static thread_local vector<Crossing> crossings;
  crossings.clear();
  crossings.reserve(candidate_surfaces.size());
  for (std::size_t i : candidate_surfaces) {
    const int32_t token = surface_tokens_[i];
    const bool coincident {std::abs(token) == std::abs(on_surface)};
    const double distance = plane_distance(i, r, u, coincident);
    if ((on_surface == 0 || distance >= FP_COINCIDENT) &&
        distance < max_distance) {
      crossings.push_back({distance, token, i, false});
    }
  }

  static thread_local vector<int8_t> boolean_values;
  boolean_values.assign(boolean_nodes_.size(), -1);
  if (on_surface != 0) {
    for (std::size_t i = 0; i < surface_tokens_.size(); ++i) {
      if (std::abs(surface_tokens_[i]) == std::abs(on_surface)) {
        set_surface_boolean_value(i, on_surface > 0, boolean_values);
        break;
      }
    }
  }
  const bool in_region = known_inside ||
                         evaluate_boolean_node(
                           boolean_root_, r, u, boolean_values);
  double total_distance {0.0};
  bool ignore_coincident {on_surface != 0};
  while (true) {
    double min_distance {INFTY};
    std::size_t nearest {crossings.size()};
    for (std::size_t i = 0; i < crossings.size(); ++i) {
      const double distance = crossings[i].distance - total_distance;
      if (crossings[i].used ||
          (ignore_coincident && distance < FP_COINCIDENT))
        continue;
      if (distance < min_distance &&
          min_distance - distance >= FP_PRECISION * min_distance) {
        min_distance = distance;
        nearest = i;
      }
    }
    if (nearest == crossings.size()) {
      return {INFTY, std::numeric_limits<int32_t>::max()};
    }

    for (std::size_t i = 0; i < crossings.size(); ++i) {
      if (i == nearest || crossings[i].used)
        continue;
      const double distance = crossings[i].distance - total_distance;
      const double tolerance = std::max(FP_COINCIDENT,
        FP_PRECISION * std::max(std::abs(min_distance), std::abs(distance)));
      if (std::abs(distance - min_distance) < tolerance) {
        return distance_complex_fallback(
          r, u, on_surface, known_inside, max_distance, &candidate_surfaces);
      }
    }

    crossings[nearest].used = true;
    total_distance = crossings[nearest].distance;
    int32_t i_surf = std::abs(crossings[nearest].token);
    if (plane_normal_projection(crossings[nearest].surface, u) <= 0.0) {
      i_surf = -i_surf;
    }
    set_surface_boolean_value(
      crossings[nearest].surface, i_surf > 0, boolean_values);
    if (evaluate_boolean_node(boolean_root_, r, u, boolean_values) !=
        in_region) {
      return {total_distance, i_surf};
    }
    ignore_coincident = true;
  }
}

//==============================================================================

bool Region::contains(Position r, Direction u, int32_t on_surface) const
{
  if (simple_) {
    return contains_simple(r, u, on_surface);
  } else {
    return contains_complex(r, u, on_surface);
  }
}

//==============================================================================

bool Region::contains_simple(Position r, Direction u, int32_t on_surface) const
{
  for (int32_t token : expression_) {
    // Assume that no tokens are operators. Evaluate the sense of particle with
    // respect to the surface and see if the token matches the sense. If the
    // particle's surface attribute is set and matches the token, that
    // overrides the determination based on sense().
    if (token == on_surface) {
    } else if (-token == on_surface) {
      return false;
    } else {
      // Note the off-by-one indexing
      bool sense = model::surfaces[abs(token) - 1]->sense(r, u);
      if (sense != (token > 0)) {
        return false;
      }
    }
  }
  return true;
}

//==============================================================================

bool Region::contains_complex(Position r, Direction u, int32_t on_surface) const
{
  bool in_cell = true;
  int total_depth = 0;

  // For each token
  for (std::size_t i = 0; i < expression_.size(); ++i) {
    int32_t token = expression_[i];

    // If the token is a surface evaluate the sense
    // If the token is a union or intersection check to
    // short circuit
    if (token < OP_UNION) {
      if (token == on_surface) {
        in_cell = true;
      } else if (-token == on_surface) {
        in_cell = false;
      } else {
        // Note the off-by-one indexing
        bool sense = model::surfaces[abs(token) - 1]->sense(r, u);
        in_cell = (sense == (token > 0));
      }
    } else if ((token == OP_UNION && in_cell == true) ||
               (token == OP_INTERSECTION && in_cell == false)) {
      // If the total depth is zero return
      if (total_depth == 0) {
        return in_cell;
      }

      total_depth--;
      i = short_circuit_jump_[i];
    } else if (token == OP_LEFT_PAREN) {
      total_depth++;
    } else if (token == OP_RIGHT_PAREN) {
      total_depth--;
    }
  }
  return in_cell;
}

//==============================================================================

BoundingBox Region::bounding_box(int32_t cell_id) const
{
  if (simple_) {
    return bounding_box_simple();
  } else {
    auto postfix = generate_postfix(cell_id);
    return bounding_box_complex(postfix);
  }
}

//==============================================================================

BoundingBox Region::bounding_box_simple() const
{
  BoundingBox bbox;
  for (int32_t token : expression_) {
    bbox &= model::surfaces[abs(token) - 1]->bounding_box(token > 0);
  }
  return bbox;
}

//==============================================================================

BoundingBox Region::bounding_box_complex(vector<int32_t> postfix) const
{
  vector<BoundingBox> stack(postfix.size());
  int i_stack = -1;

  for (auto& token : postfix) {
    if (token == OP_UNION) {
      stack[i_stack - 1] = stack[i_stack - 1] | stack[i_stack];
      i_stack--;
    } else if (token == OP_INTERSECTION) {
      stack[i_stack - 1] = stack[i_stack - 1] & stack[i_stack];
      i_stack--;
    } else {
      i_stack++;
      stack[i_stack] = model::surfaces[abs(token) - 1]->bounding_box(token > 0);
    }
  }

  assert(i_stack == 0);
  return stack.front();
}

//==============================================================================

vector<int32_t> Region::surfaces() const
{
  if (simple_) {
    return expression_;
  }

  vector<int32_t> surfaces = expression_;

  auto it = std::find_if(surfaces.begin(), surfaces.end(),
    [&](const auto& value) { return value >= OP_UNION; });

  while (it != surfaces.end()) {
    surfaces.erase(it);

    it = std::find_if(surfaces.begin(), surfaces.end(),
      [&](const auto& value) { return value >= OP_UNION; });
  }

  return surfaces;
}

//==============================================================================
// Non-method functions
//==============================================================================

void read_cells(pugi::xml_node node)
{
  // Count the number of cells.
  int n_cells = 0;
  for (pugi::xml_node cell_node : node.children("cell")) {
    n_cells++;
  }

  // Loop over XML cell elements and populate the array.
  model::cells.reserve(n_cells);
  for (pugi::xml_node cell_node : node.children("cell")) {
    model::cells.push_back(make_unique<CSGCell>(cell_node));
  }

  // Fill the cell map.
  for (int i = 0; i < model::cells.size(); i++) {
    int32_t id = model::cells[i]->id_;
    auto search = model::cell_map.find(id);
    if (search == model::cell_map.end()) {
      model::cell_map[id] = i;
    } else {
      fatal_error(
        fmt::format("Two or more cells use the same unique ID: {}", id));
    }
  }

  read_dagmc_universes(node);

  populate_universes();

  // Allocate the cell overlap count if necessary.
  if (settings::check_overlaps) {
    model::overlap_check_count.resize(model::cells.size(), 0);
  }

  if (model::cells.size() == 0) {
    fatal_error("No cells were found in the geometry.xml file");
  }
}

void populate_universes()
{
  // Used to map universe index to the index of an implicit complement cell for
  // DAGMC universes
  std::unordered_map<int, int> implicit_comp_cells;

  // Populate the Universe vector and map.
  for (int index_cell = 0; index_cell < model::cells.size(); index_cell++) {
    int32_t uid = model::cells[index_cell]->universe_;
    auto it = model::universe_map.find(uid);
    if (it == model::universe_map.end()) {
      model::universes.push_back(make_unique<Universe>());
      model::universes.back()->id_ = uid;
      model::universes.back()->cells_.push_back(index_cell);
      model::universe_map[uid] = model::universes.size() - 1;
    } else {
#ifdef OPENMC_DAGMC_ENABLED
      // Skip implicit complement cells for now
      Universe* univ = model::universes[it->second].get();
      DAGUniverse* dag_univ = dynamic_cast<DAGUniverse*>(univ);
      if (dag_univ && (dag_univ->implicit_complement_idx() == index_cell)) {
        implicit_comp_cells[it->second] = index_cell;
        continue;
      }
#endif

      model::universes[it->second]->cells_.push_back(index_cell);
    }
  }

  // Add DAGUniverse implicit complement cells last
  for (const auto& it : implicit_comp_cells) {
    int index_univ = it.first;
    int index_cell = it.second;
    model::universes[index_univ]->cells_.push_back(index_cell);
  }

  model::universes.shrink_to_fit();
}

//==============================================================================
// C-API functions
//==============================================================================

extern "C" int openmc_cell_get_fill(
  int32_t index, int* type, int32_t** indices, int32_t* n)
{
  if (index >= 0 && index < model::cells.size()) {
    Cell& c {*model::cells[index]};
    *type = static_cast<int>(c.type_);
    if (c.type_ == Fill::MATERIAL) {
      *indices = c.material_.data();
      *n = c.material_.size();
    } else {
      *indices = &c.fill_;
      *n = 1;
    }
  } else {
    set_errmsg("Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  return 0;
}

extern "C" int openmc_cell_set_fill(
  int32_t index, int type, int32_t n, const int32_t* indices)
{
  Fill filltype = static_cast<Fill>(type);
  if (index >= 0 && index < model::cells.size()) {
    Cell& c {*model::cells[index]};
    if (filltype == Fill::MATERIAL) {
      c.type_ = Fill::MATERIAL;
      c.material_.clear();
      for (int i = 0; i < n; i++) {
        int i_mat = indices[i];
        if (i_mat == MATERIAL_VOID) {
          c.material_.push_back(MATERIAL_VOID);
        } else if (i_mat >= 0 && i_mat < model::materials.size()) {
          c.material_.push_back(i_mat);
        } else {
          set_errmsg("Index in materials array is out of bounds.");
          return OPENMC_E_OUT_OF_BOUNDS;
        }
      }
      c.material_.shrink_to_fit();
    } else if (filltype == Fill::UNIVERSE) {
      c.type_ = Fill::UNIVERSE;
    } else {
      c.type_ = Fill::LATTICE;
    }
    model::geometry::mark_dirty(
      model::geometry::GeometryChange::TOPOLOGY |
      model::geometry::GeometryChange::PROPERTY);
  } else {
    set_errmsg("Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  return 0;
}

extern "C" int openmc_cell_set_temperature(
  int32_t index, double T, const int32_t* instance, bool set_contained)
{
  if (index < 0 || index >= model::cells.size()) {
    strcpy(openmc_err_msg, "Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  int32_t instance_index = instance ? *instance : -1;
  try {
    model::cells[index]->set_temperature(T, instance_index, set_contained);
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  model::geometry::mark_dirty(model::geometry::GeometryChange::PROPERTY);
  return 0;
}

extern "C" int openmc_cell_set_density(
  int32_t index, double density, const int32_t* instance, bool set_contained)
{
  if (index < 0 || index >= model::cells.size()) {
    strcpy(openmc_err_msg, "Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  int32_t instance_index = instance ? *instance : -1;
  try {
    model::cells[index]->set_density(density, instance_index, set_contained);
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  model::geometry::mark_dirty(model::geometry::GeometryChange::PROPERTY);
  return 0;
}

extern "C" int openmc_cell_get_temperature(
  int32_t index, const int32_t* instance, double* T)
{
  if (index < 0 || index >= model::cells.size()) {
    strcpy(openmc_err_msg, "Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  int32_t instance_index = instance ? *instance : -1;
  try {
    *T = model::cells[index]->temperature(instance_index);
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  return 0;
}

extern "C" int openmc_cell_get_density(
  int32_t index, const int32_t* instance, double* density)
{
  if (index < 0 || index >= model::cells.size()) {
    strcpy(openmc_err_msg, "Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  int32_t instance_index = instance ? *instance : -1;
  try {
    if (model::cells[index]->type_ != Fill::MATERIAL) {
      fatal_error(
        fmt::format("Cell {}, instance {} is not filled with a material.",
          model::cells[index]->id_, instance_index));
    }

    int32_t mat_index = model::cells[index]->material(instance_index);
    if (mat_index == MATERIAL_VOID) {
      *density = 0.0;
    } else {
      *density = model::cells[index]->density_mult(instance_index) *
                 model::materials[mat_index]->density_gpcc();
    }
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  return 0;
}

//! Get the bounding box of a cell
extern "C" int openmc_cell_bounding_box(
  const int32_t index, double* llc, double* urc)
{

  BoundingBox bbox;

  const auto& c = model::cells[index];
  bbox = c->bounding_box();

  // set lower left corner values
  llc[0] = bbox.min.x;
  llc[1] = bbox.min.y;
  llc[2] = bbox.min.z;

  // set upper right corner values
  urc[0] = bbox.max.x;
  urc[1] = bbox.max.y;
  urc[2] = bbox.max.z;

  return 0;
}

//! Get the name of a cell
extern "C" int openmc_cell_get_name(int32_t index, const char** name)
{
  if (index < 0 || index >= model::cells.size()) {
    set_errmsg("Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  *name = model::cells[index]->name().data();

  return 0;
}

//! Set the name of a cell
extern "C" int openmc_cell_set_name(int32_t index, const char* name)
{
  if (index < 0 || index >= model::cells.size()) {
    set_errmsg("Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  model::cells[index]->set_name(name);

  return 0;
}

//==============================================================================
//! Define a containing (parent) cell
//==============================================================================

//! Used to locate a universe fill in the geometry
struct ParentCell {
  bool operator==(const ParentCell& other) const
  {
    return cell_index == other.cell_index &&
           lattice_index == other.lattice_index;
  }

  bool operator<(const ParentCell& other) const
  {
    return cell_index < other.cell_index ||
           (cell_index == other.cell_index &&
             lattice_index < other.lattice_index);
  }

  int64_t cell_index;
  int64_t lattice_index;
};

//! Structure used to insert ParentCell into hashed STL data structures
struct ParentCellHash {
  std::size_t operator()(const ParentCell& p) const
  {
    return 4096 * p.cell_index + p.lattice_index;
  }
};

//! Used to manage a traversal stack when locating parent cells of a cell
//! instance in the model
struct ParentCellStack {

  //! push method that adds to the parent_cells visited cells for this search
  //! universe
  void push(int32_t search_universe, const ParentCell& pc)
  {
    parent_cells_.push_back(pc);
    // add parent cell to the set of cells we've visited for this search
    // universe
    visited_cells_[search_universe].insert(pc);
  }

  //! removes the last parent_cell and clears the visited cells for the popped
  //! cell's universe
  void pop()
  {
    visited_cells_[this->current_univ()].clear();
    parent_cells_.pop_back();
  }

  //! checks whether or not the parent cell has been visited already for this
  //! search universe
  bool visited(int32_t search_universe, const ParentCell& parent_cell)
  {
    return visited_cells_[search_universe].count(parent_cell) != 0;
  }

  //! return the next universe to search for a parent cell
  int32_t current_univ() const
  {
    return model::cells[parent_cells_.back().cell_index]->universe_;
  }

  //! indicates whether nor not parent cells are present on the stack
  bool empty() const { return parent_cells_.empty(); }

  //! compute an instance for the provided distribcell index
  int32_t compute_instance(int32_t distribcell_index) const
  {
    if (distribcell_index == C_NONE)
      return 0;

    int32_t instance = 0;
    for (const auto& parent_cell : this->parent_cells_) {
      auto& cell = model::cells[parent_cell.cell_index];
      if (cell->type_ == Fill::UNIVERSE) {
        instance += cell->offset_[distribcell_index];
      } else if (cell->type_ == Fill::LATTICE) {
        auto& lattice = model::lattices[cell->fill_];
        instance +=
          lattice->offset(distribcell_index, parent_cell.lattice_index);
      }
    }
    return instance;
  }

  // Accessors
  vector<ParentCell>& parent_cells() { return parent_cells_; }
  const vector<ParentCell>& parent_cells() const { return parent_cells_; }

  // Data Members
  vector<ParentCell> parent_cells_;
  std::unordered_map<int32_t, std::unordered_set<ParentCell, ParentCellHash>>
    visited_cells_;
};

vector<ParentCell> Cell::find_parent_cells(
  int32_t instance, const Position& r) const
{

  // create a temporary particle
  GeometryState dummy_particle {};
  dummy_particle.r() = r;
  dummy_particle.u() = {0., 0., 1.};

  return find_parent_cells(instance, dummy_particle);
}

vector<ParentCell> Cell::find_parent_cells(
  int32_t instance, GeometryState& p) const
{
  // look up the particle's location
  exhaustive_find_cell(p);
  const auto& coords = p.coord();

  // build a parent cell stack from the particle coordinates
  ParentCellStack stack;
  bool cell_found = false;
  for (auto it = coords.begin(); it != coords.end(); it++) {
    const auto& coord = *it;
    const auto& cell = model::cells[coord.cell()];
    // if the cell at this level matches the current cell, stop adding to the
    // stack
    if (coord.cell() == model::cell_map[this->id_]) {
      cell_found = true;
      break;
    }

    // if filled with a lattice, get the lattice index from the next
    // level in the coordinates to push to the stack
    int lattice_idx = C_NONE;
    if (cell->type_ == Fill::LATTICE) {
      const auto& next_coord = *(it + 1);
      lattice_idx = model::lattices[next_coord.lattice()]->get_flat_index(
        next_coord.lattice_index());
    }
    stack.push(coord.universe(), {coord.cell(), lattice_idx});
  }

  // if this loop finished because the cell was found and
  // the instance matches the one requested in the call
  // we have the correct path and can return the stack
  if (cell_found &&
      stack.compute_instance(this->distribcell_index_) == instance) {
    return stack.parent_cells();
  }

  // fall back on an exhaustive search for the cell's parents
  return exhaustive_find_parent_cells(instance);
}

vector<ParentCell> Cell::exhaustive_find_parent_cells(int32_t instance) const
{
  ParentCellStack stack;
  // start with this cell's universe
  int32_t prev_univ_idx;
  int32_t univ_idx = this->universe_;

  while (true) {
    const auto& univ = model::universes[univ_idx];
    prev_univ_idx = univ_idx;

    // search for a cell that is filled w/ this universe
    for (const auto& cell : model::cells) {
      // if this is a material-filled cell, move on
      if (cell->type_ == Fill::MATERIAL)
        continue;

      if (cell->type_ == Fill::UNIVERSE) {
        // if this is in the set of cells previously visited for this universe,
        // move on
        if (stack.visited(univ_idx, {model::cell_map[cell->id_], C_NONE}))
          continue;

        // if this cell contains the universe we're searching for, add it to the
        // stack
        if (cell->fill_ == univ_idx) {
          stack.push(univ_idx, {model::cell_map[cell->id_], C_NONE});
          univ_idx = cell->universe_;
        }
      } else if (cell->type_ == Fill::LATTICE) {
        // retrieve the lattice and lattice universes
        const auto& lattice = model::lattices[cell->fill_];
        const auto& lattice_univs = lattice->universes_;

        // start search for universe
        auto lat_it = lattice_univs.begin();
        while (true) {
          // find the next lattice cell with this universe
          lat_it = std::find(lat_it, lattice_univs.end(), univ_idx);
          if (lat_it == lattice_univs.end())
            break;

          int lattice_idx = lat_it - lattice_univs.begin();

          // move iterator forward one to avoid finding the same entry
          lat_it++;
          if (stack.visited(
                univ_idx, {model::cell_map[cell->id_], lattice_idx}))
            continue;

          // add this cell and lattice index to the stack and exit loop
          stack.push(univ_idx, {model::cell_map[cell->id_], lattice_idx});
          univ_idx = cell->universe_;
          break;
        }
      }
      // if we've updated the universe, break
      if (prev_univ_idx != univ_idx)
        break;
    } // end cell loop search for universe

    // if we're at the top of the geometry and the instance matches, we're done
    if (univ_idx == model::root_universe &&
        stack.compute_instance(this->distribcell_index_) == instance)
      break;

    // if there is no match on the original cell's universe, report an error
    if (univ_idx == this->universe_) {
      fatal_error(
        fmt::format("Could not find the parent cells for cell {}, instance {}.",
          this->id_, instance));
    }

    // if we don't find a suitable update, adjust the stack and continue
    if (univ_idx == model::root_universe || univ_idx == prev_univ_idx) {
      stack.pop();
      univ_idx = stack.empty() ? this->universe_ : stack.current_univ();
    }

  } // end while

  // reverse the stack so the highest cell comes first
  std::reverse(stack.parent_cells().begin(), stack.parent_cells().end());
  return stack.parent_cells();
}

std::unordered_map<int32_t, vector<int32_t>> Cell::get_contained_cells(
  int32_t instance, Position* hint) const
{
  std::unordered_map<int32_t, vector<int32_t>> contained_cells;

  // if this is a material-filled cell it has no contained cells
  if (this->type_ == Fill::MATERIAL)
    return contained_cells;

  // find the pathway through the geometry to this cell
  vector<ParentCell> parent_cells;

  // if a positional hint is provided, attempt to do a fast lookup
  // of the parent cells
  parent_cells = hint ? find_parent_cells(instance, *hint)
                      : exhaustive_find_parent_cells(instance);

  // if this cell is filled w/ a material, it contains no other cells
  if (type_ != Fill::MATERIAL) {
    this->get_contained_cells_inner(contained_cells, parent_cells);
  }

  return contained_cells;
}

//! Get all cells within this cell
void Cell::get_contained_cells_inner(
  std::unordered_map<int32_t, vector<int32_t>>& contained_cells,
  vector<ParentCell>& parent_cells) const
{

  // filled by material, determine instance based on parent cells
  if (type_ == Fill::MATERIAL) {
    int instance = 0;
    if (this->distribcell_index_ >= 0) {
      for (auto& parent_cell : parent_cells) {
        auto& cell = model::cells[parent_cell.cell_index];
        if (cell->type_ == Fill::UNIVERSE) {
          instance += cell->offset_[distribcell_index_];
        } else if (cell->type_ == Fill::LATTICE) {
          auto& lattice = model::lattices[cell->fill_];
          instance += lattice->offset(
            this->distribcell_index_, parent_cell.lattice_index);
        }
      }
    }
    // add entry to contained cells
    contained_cells[model::cell_map[id_]].push_back(instance);
    // filled with universe, add the containing cell to the parent cells
    // and recurse
  } else if (type_ == Fill::UNIVERSE) {
    parent_cells.push_back({model::cell_map[id_], -1});
    auto& univ = model::universes[fill_];
    for (auto cell_index : univ->cells_) {
      auto& cell = model::cells[cell_index];
      cell->get_contained_cells_inner(contained_cells, parent_cells);
    }
    parent_cells.pop_back();
    // filled with a lattice, visit each universe in the lattice
    // with a recursive call to collect the cell instances
  } else if (type_ == Fill::LATTICE) {
    auto& lattice = model::lattices[fill_];
    for (auto i = lattice->begin(); i != lattice->end(); ++i) {
      auto& univ = model::universes[*i];
      parent_cells.push_back({model::cell_map[id_], i.indx_});
      for (auto cell_index : univ->cells_) {
        auto& cell = model::cells[cell_index];
        cell->get_contained_cells_inner(contained_cells, parent_cells);
      }
      parent_cells.pop_back();
    }
  }
}

//! Return the index in the cells array of a cell with a given ID
extern "C" int openmc_get_cell_index(int32_t id, int32_t* index)
{
  auto it = model::cell_map.find(id);
  if (it != model::cell_map.end()) {
    *index = it->second;
    return 0;
  } else {
    set_errmsg("No cell exists with ID=" + std::to_string(id) + ".");
    return OPENMC_E_INVALID_ID;
  }
}

//! Return the ID of a cell
extern "C" int openmc_cell_get_id(int32_t index, int32_t* id)
{
  if (index >= 0 && index < model::cells.size()) {
    *id = model::cells[index]->id_;
    return 0;
  } else {
    set_errmsg("Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

//! Set the ID of a cell
extern "C" int openmc_cell_set_id(int32_t index, int32_t id)
{
  if (index >= 0 && index < model::cells.size()) {
    const int32_t old_id = model::cells[index]->id_;
    auto existing = model::cell_map.find(id);
    if (existing != model::cell_map.end() && existing->second != index) {
      set_errmsg("Cell ID=" + std::to_string(id) + " is already in use.");
      return OPENMC_E_INVALID_ID;
    }
    model::cell_map.erase(old_id);
    model::cells[index]->id_ = id;
    model::cell_map[id] = index;
    model::geometry::mark_dirty(model::geometry::GeometryChange::IDENTITY);
    return 0;
  } else {
    set_errmsg("Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

//! Return the translation vector of a cell
extern "C" int openmc_cell_get_translation(int32_t index, double xyz[])
{
  if (index >= 0 && index < model::cells.size()) {
    auto& cell = model::cells[index];
    xyz[0] = cell->translation_.x;
    xyz[1] = cell->translation_.y;
    xyz[2] = cell->translation_.z;
    return 0;
  } else {
    set_errmsg("Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

//! Set the translation vector of a cell
extern "C" int openmc_cell_set_translation(int32_t index, const double xyz[])
{
  if (index >= 0 && index < model::cells.size()) {
    if (model::cells[index]->fill_ == C_NONE) {
      set_errmsg(fmt::format("Cannot apply a translation to cell {}"
                             " because it is not filled with another universe",
        index));
      return OPENMC_E_GEOMETRY;
    }
    model::cells[index]->translation_ = Position(xyz);
    model::geometry::mark_dirty(model::geometry::GeometryChange::SPATIAL);
    return 0;
  } else {
    set_errmsg("Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

//! Return the rotation matrix of a cell
extern "C" int openmc_cell_get_rotation(int32_t index, double rot[], size_t* n)
{
  if (index >= 0 && index < model::cells.size()) {
    auto& cell = model::cells[index];
    *n = cell->rotation_.size();
    std::memcpy(rot, cell->rotation_.data(), *n * sizeof(cell->rotation_[0]));
    return 0;
  } else {
    set_errmsg("Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

//! Set the flattened rotation matrix of a cell
extern "C" int openmc_cell_set_rotation(
  int32_t index, const double rot[], size_t rot_len)
{
  if (index >= 0 && index < model::cells.size()) {
    if (model::cells[index]->fill_ == C_NONE) {
      set_errmsg(fmt::format("Cannot apply a rotation to cell {}"
                             " because it is not filled with another universe",
        index));
      return OPENMC_E_GEOMETRY;
    }
    std::vector<double> vec_rot(rot, rot + rot_len);
    model::cells[index]->set_rotation(vec_rot);
    model::geometry::mark_dirty(model::geometry::GeometryChange::SPATIAL);
    return 0;
  } else {
    set_errmsg("Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

//! Get the number of instances of the requested cell
extern "C" int openmc_cell_get_num_instances(
  int32_t index, int32_t* num_instances)
{
  if (index < 0 || index >= model::cells.size()) {
    set_errmsg("Index in cells array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  *num_instances = model::cells[index]->n_instances();
  return 0;
}

//! Extend the cells array by n elements
extern "C" int openmc_extend_cells(
  int32_t n, int32_t* index_start, int32_t* index_end)
{
  if (index_start)
    *index_start = model::cells.size();
  if (index_end)
    *index_end = model::cells.size() + n - 1;
  for (int32_t i = 0; i < n; i++) {
    model::cells.push_back(make_unique<CSGCell>());
  }
  if (n > 0) {
    model::geometry::mark_dirty(
      model::geometry::GeometryChange::TOPOLOGY |
      model::geometry::GeometryChange::IDENTITY);
  }
  return 0;
}

extern "C" int cells_size()
{
  return model::cells.size();
}

} // namespace openmc
