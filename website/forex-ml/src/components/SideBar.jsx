import { Link } from "react-router-dom";
import "./SideBar.css";
function SideBar() {
  return (
    <aside className="sidebar">
      <h2 className="sidebar-title">Maching Learning w/ Foreign Exchange</h2>
      <p className="sidebar-description">Explore the different sections here!</p>
      <nav>
        <ul className="sidebar-list">
          <li>
            <Link to='/' className="sidebar-link">
                  Home
            </Link>
            <Link to='/misleading' className="sidebar-link">
                The Misleading Graphs
            </Link>
          </li>
        </ul>
      </nav>
    </aside>
  );
}

export default SideBar;
