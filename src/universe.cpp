#include "openmc/universe.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "openmc/hdf5_interface.h"
#include "openmc/model/geometry/compiled_geometry.h"
#include "openmc/particle.h"

namespace openmc {

namespace model {

std::unordered_map<int32_t, int32_t> universe_map;
vector<unique_ptr<Universe>> universes;

} // namespace model

//==============================================================================
// Universe implementation
//==============================================================================

void Universe::to_hdf5(hid_t universes_group) const
{
  // Create a group for this universe.
  auto group = create_group(universes_group, fmt::format("universe {}", id_));

  // Write the geometry representation type.
  write_string(group, "geom_type", "csg", false);

  // Write the contained cells.
  if (cells_.size() > 0) {
    vector<int32_t> cell_ids;
    for (auto i_cell : cells_)
      cell_ids.push_back(model::cells[i_cell]->id_);
    write_dataset(group, "cells", cell_ids);
  }

  close_group(group);
}

bool Universe::find_cell(GeometryState& p) const
{
  Position r {p.r_local()};
  Position u {p.u_local()};
  auto surf = p.surface();
  int32_t i_univ = p.lowest_coord().universe();

  auto search = [&](const vector<int32_t>& cells) {
    for (auto i_cell : cells) {
      if (model::cells[i_cell]->universe_ != i_univ)
        continue;
      // Bounding boxes only select candidates. Exact region containment remains
      // authoritative.
      if (model::cells[i_cell]->contains(r, u, surf)) {
        p.lowest_coord().cell() = i_cell;
        return true;
      }
    }
    return false;
  };

  // A known surface can override its numerically evaluated sense, so an AABB
  // based only on the coordinates is not sufficient for those searches.
  if (!partitioner_ || surf != SURFACE_NONE ||
      !model::geometry::is_spatial_current())
    return search(cells_);

  vector<int32_t> candidates;
  partitioner_->get_cells(r, candidates);
  if (search(candidates)) {
    const int32_t candidate_match = p.lowest_coord().cell();
    std::size_t candidate = 0;
    for (auto i_cell : cells_) {
      if (i_cell == candidate_match)
        break;
      if (candidate < candidates.size() && candidates[candidate] == i_cell) {
        ++candidate;
        continue;
      }
      if (model::cells[i_cell]->universe_ == i_univ &&
          model::cells[i_cell]->contains(r, u, surf)) {
        p.lowest_coord().cell() = i_cell;
        return true;
      }
    }
    p.lowest_coord().cell() = candidate_match;
    return true;
  }

  // Keep an exhaustive fallback so an unexpectedly conservative bound can
  // never turn a successful point-location query into a lost particle.
  if (candidates.size() < cells_.size())
    return search(cells_);

  return false;
}

namespace {

bool contains(const BoundingBox& box, Position r)
{
  return r.x >= box.min.x && r.x <= box.max.x && r.y >= box.min.y &&
         r.y <= box.max.y && r.z >= box.min.z && r.z <= box.max.z;
}

bool finite_valid(const BoundingBox& box)
{
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(box.min[i]) || !std::isfinite(box.max[i]) ||
        box.min[i] <= -INFTY || box.max[i] >= INFTY ||
        box.min[i] > box.max[i]) {
      return false;
    }
  }
  return true;
}

} // namespace

BoundingBox Universe::bounding_box() const
{
  BoundingBox bbox = BoundingBox::inverted();
  if (cells_.size() == 0) {
    return {};
  } else {
    for (const auto& cell : cells_) {
      auto& c = model::cells[cell];
      bbox |= c->bounding_box();
    }
  }
  return bbox;
}

//==============================================================================
// UniversePartitioner implementation
//==============================================================================

UniversePartitioner::UniversePartitioner(const Universe& univ)
{
  bounded_.reserve(univ.cells_.size());
  spill_.reserve(univ.cells_.size());
  for (int32_t order = 0; order < univ.cells_.size(); ++order) {
    const int32_t i_cell = univ.cells_[order];
    BoundingBox box = model::cells[i_cell]->bounding_box();
    CellBounds item {box, i_cell, order};
    if (finite_valid(box)) {
      // Include the adjacent floating-point values to avoid rejecting points
      // because a mathematically exact boundary rounded inward.
      for (int axis = 0; axis < 3; ++axis) {
        item.box.min[axis] = std::nextafter(
          item.box.min[axis] - FP_COINCIDENT, -INFINITY);
        item.box.max[axis] = std::nextafter(
          item.box.max[axis] + FP_COINCIDENT, INFINITY);
      }
      bounded_.push_back(item);
    } else {
      spill_.push_back(item);
    }
  }

  if (!bounded_.empty())
    build(0, bounded_.size());
}

int32_t UniversePartitioner::build(int32_t begin, int32_t end)
{
  const int32_t node_index = nodes_.size();
  nodes_.push_back({});
  auto& box = nodes_[node_index].box;
  box = BoundingBox::inverted();
  for (int32_t i = begin; i < end; ++i)
    box |= bounded_[i].box;

  constexpr int LEAF_SIZE = 4;
  if (end - begin <= LEAF_SIZE) {
    nodes_[node_index].begin = begin;
    nodes_[node_index].end = end;
    return node_index;
  }

  Position centroid_min {INFTY, INFTY, INFTY};
  Position centroid_max {-INFTY, -INFTY, -INFTY};
  for (int32_t i = begin; i < end; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const double center = bounded_[i].box.min[axis] +
                            0.5 * (bounded_[i].box.max[axis] -
                                    bounded_[i].box.min[axis]);
      centroid_min[axis] = std::min(centroid_min[axis], center);
      centroid_max[axis] = std::max(centroid_max[axis], center);
    }
  }

  int axis = 0;
  if (centroid_max.y - centroid_min.y > centroid_max.x - centroid_min.x)
    axis = 1;
  if (centroid_max.z - centroid_min.z >
      centroid_max[axis] - centroid_min[axis])
    axis = 2;

  const int32_t middle = begin + (end - begin) / 2;
  std::nth_element(bounded_.begin() + begin, bounded_.begin() + middle,
    bounded_.begin() + end, [axis](const CellBounds& a, const CellBounds& b) {
      const double a_center =
        a.box.min[axis] + 0.5 * (a.box.max[axis] - a.box.min[axis]);
      const double b_center =
        b.box.min[axis] + 0.5 * (b.box.max[axis] - b.box.min[axis]);
      return a_center == b_center ? a.order < b.order : a_center < b_center;
    });

  const int32_t left = build(begin, middle);
  const int32_t right = build(middle, end);
  nodes_[node_index].left = left;
  nodes_[node_index].right = right;
  return node_index;
}

void UniversePartitioner::query(
  int32_t node_index, Position r, vector<CellBounds>& hits) const
{
  const auto& node = nodes_[node_index];
  if (!contains(node.box, r))
    return;

  if (node.left < 0) {
    for (int32_t i = node.begin; i < node.end; ++i) {
      if (contains(bounded_[i].box, r))
        hits.push_back(bounded_[i]);
    }
  } else {
    query(node.left, r, hits);
    query(node.right, r, hits);
  }
}

void UniversePartitioner::get_cells(Position r, vector<int32_t>& cells) const
{
  vector<CellBounds> hits {spill_};
  if (!nodes_.empty())
    query(0, r, hits);

  std::sort(hits.begin(), hits.end(),
    [](const CellBounds& a, const CellBounds& b) { return a.order < b.order; });
  cells.reserve(cells.size() + hits.size());
  for (const auto& hit : hits)
    cells.push_back(hit.cell);
}

} // namespace openmc
