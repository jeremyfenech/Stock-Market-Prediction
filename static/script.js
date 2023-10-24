// Set the tables to DataTables
$(document).ready(function () {
  $("#top-gainers-table").DataTable({
    order: [[4, "desc"]], 
  });
  $("#top-losers-table").DataTable({
    order: [[4, "asc"]], 
  });
  $("#index-table").DataTable({
    order: [[4, "desc"]], 
    iDisplayLength: 50
  });
});

// Adjust the size of the navbar when scrolling
window.addEventListener("scroll", function () {
  var navbar = document.getElementById("navbar");
  if (window.scrollY > 50) {
    navbar.classList.add("navbar-shrink");
  } else {
    navbar.classList.remove("navbar-shrink");
  }
});
