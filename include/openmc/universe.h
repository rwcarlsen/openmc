#ifndef OPENMC_UNIVERSE_H
#define OPENMC_UNIVERSE_H

#include "openmc/bounding_box.h"
#include "openmc/cell.h"

namespace openmc {

#ifdef OPENMC_DAGMC_ENABLED
class DAGUniverse;
#endif

class GeometryState;
class Universe;
class UniversePartitioner;

namespace model {

extern std::unordered_map<int32_t, int32_t> universe_map;
extern vector<unique_ptr<Universe>> universes;

} // namespace model

//==============================================================================
//! A geometry primitive that fills all space and contains cells.
//==============================================================================

class Universe {
public:
  int32_t id_;            //!< Unique ID
  vector<int32_t> cells_; //!< Cells within this universe
  int32_t n_instances_;   //!< Number of instances of this universe

  //! \brief Write universe information to an HDF5 group.
  //! \param group_id An HDF5 group id.
  virtual void to_hdf5(hid_t group_id) const;

  virtual bool find_cell(GeometryState& p) const;

  BoundingBox bounding_box() const;

  /* By default, universes are CSG universes. The DAGMC
   * universe overrides standard behaviors, and in the future,
   * other things might too.
   */
  virtual GeometryType geom_type() const { return GeometryType::CSG; }

  unique_ptr<UniversePartitioner> partitioner_;
};

//==============================================================================
//! Speeds up geometry searches using finite cell bounding boxes.
//==============================================================================

class UniversePartitioner {
public:
  explicit UniversePartitioner(const Universe& univ);

  //! Append cells whose bounds could contain a point, in universe order.
  void get_cells(Position r, vector<int32_t>& cells) const;

  //! Whether bounded cells make candidate lookup preferable to linear search.
  bool useful() const
  {
    return bounded_.size() >= 4 && bounded_.size() > spill_.size();
  }

private:
  struct CellBounds {
    BoundingBox box;
    int32_t cell;
    int32_t order;
  };

  struct Node {
    BoundingBox box;
    int32_t left {-1};
    int32_t right {-1};
    int32_t begin {0};
    int32_t end {0};
  };

  int32_t build(int32_t begin, int32_t end);
  void query(int32_t node, Position r, vector<CellBounds>& hits) const;

  vector<CellBounds> bounded_;
  vector<CellBounds> spill_;
  vector<Node> nodes_;
};

} // namespace openmc
#endif // OPENMC_UNIVERSE_H
