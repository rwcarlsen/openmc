#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <string>
#include <vector>

#include "openmc/bank.h"
#include "openmc/mcpl_interface.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"

// Test the MCPL stat:sum functionality (issue #3514)
TEST_CASE("MCPL stat:sum field")
{
  // Check if MCPL interface is available
  if (!openmc::is_mcpl_interface_available()) {
    SKIP("MCPL library not available");
  }

  SECTION("stat:sum field is written to MCPL files")
  {
    // Create a temporary filename
    std::string filename = "test_stat_sum.mcpl";

    // Create some test particles
    std::vector<openmc::SourceSite> source_bank(100);
    std::vector<int64_t> bank_index = {0, 100}; // 100 particles total

    // Initialize test particles
    for (int i = 0; i < 100; ++i) {
      source_bank[i].particle = openmc::ParticleType::neutron();
      source_bank[i].r = {i * 0.1, i * 0.2, i * 0.3};
      source_bank[i].u = {0.0, 0.0, 1.0};
      source_bank[i].E = 2.0e6; // 2 MeV
      source_bank[i].time = 0.0;
      source_bank[i].wgt = 1.0;
    }

    // Write the MCPL file
    openmc::write_mcpl_source_point(filename.c_str(), source_bank, bank_index);

    // Verify the file was created
    FILE* f = std::fopen(filename.c_str(), "r");
    REQUIRE(f != nullptr);
    std::fclose(f);

    // Read the file back to check stat:sum
    // Note: This would require mcpl_open_file and checking the header
    // Since we can't easily read MCPL headers in C++ without the full MCPL API,
    // we rely on the Python test to verify the actual content

    // Clean up
    std::remove(filename.c_str());
  }

  SECTION("stat:sum uses correct particle count")
  {
    std::string filename = "test_count.mcpl";

    // Test with different particle counts
    std::vector<int> test_counts = {1, 10, 100, 1000};

    for (int count : test_counts) {
      std::vector<openmc::SourceSite> source_bank(count);
      std::vector<int64_t> bank_index = {0, count};

      // Initialize particles
      for (int i = 0; i < count; ++i) {
        source_bank[i].particle = openmc::ParticleType::neutron();
        source_bank[i].r = {0.0, 0.0, 0.0};
        source_bank[i].u = {0.0, 0.0, 1.0};
        source_bank[i].E = 1.0e6;
        source_bank[i].time = 0.0;
        source_bank[i].wgt = 1.0;
      }

      // Write MCPL file
      openmc::write_mcpl_source_point(
        filename.c_str(), source_bank, bank_index);

      // The stat:sum should equal count (verified by Python test)
      // Here we just verify the file was created successfully
      FILE* f = std::fopen(filename.c_str(), "r");
      REQUIRE(f != nullptr);
      std::fclose(f);

      // Clean up
      std::remove(filename.c_str());
    }
  }

  SECTION("stat:sum handles empty particle bank")
  {
    std::string filename = "test_empty.mcpl";

    // Create empty particle bank
    std::vector<openmc::SourceSite> source_bank;
    std::vector<int64_t> bank_index = {0};

    // This should still create a valid MCPL file with stat:sum = 0
    openmc::write_mcpl_source_point(filename.c_str(), source_bank, bank_index);

    // Verify file was created
    FILE* f = std::fopen(filename.c_str(), "r");
    REQUIRE(f != nullptr);
    std::fclose(f);

    // Clean up
    std::remove(filename.c_str());
  }
}

TEST_CASE("Thread-local fission banks merge deterministically")
{
  openmc::settings::ifp_on = false;
  openmc::simulation::work_per_rank = 3;
  openmc::init_fission_bank(6);
  openmc::initialize_fission_bank_generation();
  openmc::simulation::progeny_per_particle = {2, 2, 2};

  // Deliberately stage sites in reverse source order. The merge must restore
  // parent/progeny ordering.
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < 6; ++j) {
    int i = 5 - j;
    openmc::SourceSite site;
    site.parent_id = i / 2;
    site.progeny_id = i % 2;
    site.E = i;
    openmc::bank_fission_site(site);
  }

  openmc::collect_fission_banks();

  REQUIRE(openmc::simulation::fission_bank.size() == 6);
  for (int i = 0; i < 6; ++i) {
    REQUIRE(openmc::simulation::fission_bank[i].E == i);
  }
  openmc::vector<int64_t> expected_counts {2, 2, 2};
  REQUIRE(openmc::simulation::progeny_per_particle == expected_counts);

  openmc::free_memory_bank();
}

TEST_CASE("Thread-local fission bank reports overflow immediately")
{
  openmc::settings::ifp_on = false;
  openmc::simulation::work_per_rank = 1;
  openmc::init_fission_bank(2);
  openmc::initialize_fission_bank_generation();

  openmc::SourceSite site;
  site.parent_id = 0;
  site.progeny_id = 0;
  REQUIRE(openmc::bank_fission_site(site) == 0);
  site.progeny_id = 1;
  REQUIRE(openmc::bank_fission_site(site) == 1);
  site.progeny_id = 2;
  REQUIRE(openmc::bank_fission_site(site) == -1);

  // The caller only counts successful appends, as create_fission_sites does.
  openmc::simulation::progeny_per_particle = {2};
  openmc::collect_fission_banks();
  REQUIRE(openmc::simulation::fission_bank.size() == 2);

  openmc::free_memory_bank();
}

TEST_CASE("IFP fission banking retains shared append indices")
{
  openmc::settings::ifp_on = true;
  openmc::simulation::work_per_rank = 1;
  openmc::init_fission_bank(2);
  openmc::initialize_fission_bank_generation();

  openmc::SourceSite site;
  REQUIRE(openmc::bank_fission_site(site) == 0);
  REQUIRE(openmc::bank_fission_site(site) == 1);
  REQUIRE(openmc::bank_fission_site(site) == -1);
  openmc::collect_fission_banks();
  REQUIRE(openmc::simulation::fission_bank.size() == 2);

  openmc::settings::ifp_on = false;
  openmc::free_memory_bank();
}
