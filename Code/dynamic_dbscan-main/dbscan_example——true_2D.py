"""
Enhanced dynamic DBSCAN clustering algorithm with real-world map visualization.
Visualizes clustering results on actual geographic coordinates using Folium.
Fixed issues: 1) Brighter map background 2) Output to organized folder 3) Fixed plot overlapping
Modified to use real truck trajectory data instead of simulated Chengdu coordinates.
"""
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium import plugins
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from datetime import datetime
import json
import os
from experiments import enhanced_adaptive_dbscan_params
# Import your dynamic DBSCAN module
# import dbscan.dynamic_fdbscan
# Import the trajectory dataset module
import alglab.dataset_base

class MockDynamicDBSCAN:
    """Mock implementation for demonstration purposes"""
    def __init__(self, eps, min_samples, alpha, beta, initial_data):
        from sklearn.cluster import DBSCAN
        self.eps = eps
        self.min_samples = min_samples
        self.data_points = list(initial_data)
        self.current_labels = None
        self._recluster()
    
    def _recluster(self):
        if len(self.data_points) < self.min_samples:
            self.current_labels = [-1] * len(self.data_points)
            return
        
        from sklearn.cluster import DBSCAN
        # Scale eps based on coordinate system (degrees)
        scaled_eps = self.eps # Adjust for geographic coordinates
        dbscan = DBSCAN(eps=scaled_eps, min_samples=self.min_samples)
        self.current_labels = dbscan.fit_predict(np.array(self.data_points))
    
    def add_point(self, point):
        self.data_points.append(point)
        self._recluster()
    
    def get_cluster(self, index):
        if index < len(self.current_labels):
            return self.current_labels[index]
        return -1

def create_output_directory():
    """Create organized output directory structure"""
    # Create main output directory
    base_dir = "dynamic_dbscan-main"
    output_dir = os.path.join(base_dir, "clustering_results")
    
    # Create directories if they don't exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def load_truck_trajectory_data(base_path="E:/dwd_truck", dates=["2024-04-14"], n_samples=None):
    """
    Load real truck trajectory data from the specified dataset
    
    Args:
        base_path: Base path to the truck dataset
        dates: List of dates to load data from
        n_samples: Maximum number of samples to use (None for all data)
    
    Returns:
        coordinates: numpy array of [longitude, latitude] coordinates
        additional_info: dictionary with metadata about the dataset
    """
    try:
        print(f"Loading truck trajectory data from {base_path}")
        print(f"Date range: {dates}")
        
        # Load the trajectory dataset
        truck_dataset = alglab.dataset_base.LocalTrajectoryDataset(
            base_folder=base_path, 
            date_range=dates
        )
        
        # Get the data
        data = truck_dataset.data
        print(f"Loaded dataset with {len(data)} records")
        
        # Extract coordinates based on the data structure
        # Assuming the data has longitude and latitude columns
        # You may need to adjust these column names based on your actual data structure
        if hasattr(data, 'columns'):
            # If it's a pandas DataFrame
            print(f"Data columns: {list(data.columns)}")
            
            # Try common coordinate column names
            lon_cols = ['longitude', 'lon', 'lng', 'x', 'Longitude', 'LON']
            lat_cols = ['latitude', 'lat', 'y', 'Latitude', 'LAT']
            
            lon_col = None
            lat_col = None
            
            for col in lon_cols:
                if col in data.columns:
                    lon_col = col
                    break
            
            for col in lat_cols:
                if col in data.columns:
                    lat_col = col
                    break
            
            if lon_col is None or lat_col is None:
                raise ValueError(f"Could not find longitude/latitude columns in data. Available columns: {list(data.columns)}")
            
            print(f"Using longitude column: {lon_col}")
            print(f"Using latitude column: {lat_col}")
            
            # Extract coordinates
            coordinates = data[[lon_col, lat_col]].values
            
        else:
            # If it's a different data structure, try to extract coordinates
            # This might need to be adjusted based on your actual data format
            print(f"Data type: {type(data)}")
            if hasattr(data, '__len__'):
                print(f"Data length: {len(data)}")
            
            # Try to convert to numpy array assuming it's already coordinate pairs
            coordinates = np.array(data)
            if coordinates.shape[1] != 2:
                raise ValueError(f"Expected 2D coordinate data, got shape: {coordinates.shape}")
        
        # Remove any invalid coordinates (NaN, inf, etc.)
        valid_mask = np.isfinite(coordinates).all(axis=1)
        coordinates = coordinates[valid_mask]
        
        print(f"Valid coordinates after filtering: {len(coordinates)}")
        
        # Sample data if requested
        if n_samples is not None and len(coordinates) > n_samples:
            # Use random sampling to get a representative subset
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(len(coordinates), n_samples, replace=False)
            coordinates = coordinates[sample_indices]
            print(f"Sampled down to {n_samples} points")
        
        # Calculate data statistics
        lon_min, lon_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        lat_min, lat_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        
        print(f"Longitude range: [{lon_min:.6f}, {lon_max:.6f}]")
        print(f"Latitude range: [{lat_min:.6f}, {lat_max:.6f}]")
        
        # Prepare additional information
        additional_info = {
            'source': 'Real truck trajectory data',
            'base_path': base_path,
            'dates': dates,
            'total_points': len(coordinates),
            'longitude_range': (lon_min, lon_max),
            'latitude_range': (lat_min, lat_max),
            'center_longitude': np.mean(coordinates[:, 0]),
            'center_latitude': np.mean(coordinates[:, 1])
        }
        
        return coordinates, additional_info
        
    except Exception as e:
        print(f"Error loading truck trajectory data: {e}")
        print("Falling back to generating sample data...")
        
        # Fallback: generate some sample data for demonstration
        np.random.seed(42)
        n_points = n_samples if n_samples else 200
        
        # Generate coordinates in a reasonable geographic range
        # Assuming the truck data might be in China, use approximate bounds
        lon_center, lat_center = 116.4, 39.9  # Beijing area as example
        lon_spread, lat_spread = 2.0, 1.5
        
        coordinates = np.random.normal(
            [lon_center, lat_center], 
            [lon_spread, lat_spread], 
            (n_points, 2)
        )
        
        additional_info = {
            'source': 'Fallback sample data',
            'base_path': base_path,
            'dates': dates,
            'total_points': len(coordinates),
            'longitude_range': (coordinates[:, 0].min(), coordinates[:, 0].max()),
            'latitude_range': (coordinates[:, 1].min(), coordinates[:, 1].max()),
            'center_longitude': np.mean(coordinates[:, 0]),
            'center_latitude': np.mean(coordinates[:, 1])
        }
        
        return coordinates, additional_info

def create_interactive_map(data, labels, data_info, output_dir, save_path="truck_dbscan_clusters.html"):
    """Create an interactive Folium map with bright background"""
    
    # Calculate map center from data info
    center_lat = data_info['center_latitude']
    center_lon = data_info['center_longitude']
    
    # Create base map with bright tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,  # Adjust zoom based on data spread
        tiles='OpenStreetMap'  # Default bright tile
    )
    
    # Add bright alternative tile layers
    folium.TileLayer('cartodbpositron', name='CartoDB Positron (Bright)').add_to(m)
    
    # Add custom tile layers with proper attribution
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.',
        name='Terrain View'
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.',
        name='Light Theme'
    ).add_to(m)
    
    # Define bright colors for clusters
    unique_labels = np.unique(labels)
    bright_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                     'lightred', 'darkblue', 'darkgreen', 'cadetblue', 
                     'pink', 'lightblue', 'lightgreen', 'beige']
    
    # Create cluster groups for layer control
    cluster_groups = {}
    
    for label in unique_labels:
        if label == -1:
            group_name = "Noise Points"
            color = 'gray'
        else:
            group_name = f"Cluster {label}"
            color = bright_colors[label % len(bright_colors)]
        
        cluster_groups[label] = folium.FeatureGroup(name=group_name)
        
        # Add points for this cluster
        cluster_mask = labels == label
        cluster_data = data[cluster_mask]
        
        for i, (lon, lat) in enumerate(cluster_data):
            if label == -1:
                # Noise points as small circles without markers
                folium.CircleMarker(
                    [lat, lon],
                    radius=3,
                    color='gray',
                    fillColor='gray',
                    fillOpacity=0.6,
                    weight=1
                ).add_to(cluster_groups[label])
            else:
                # Cluster points as small circles with bright colors
                folium.CircleMarker(
                    [lat, lon],
                    radius=4,
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=1
                ).add_to(cluster_groups[label])
    
    # Add all cluster groups to map
    for group in cluster_groups.values():
        group.add_to(m)
    
    # Add heat map layer
    heat_data = [[row[1], row[0]] for row in data]  # lat, lon format for heatmap
    heat_map = plugins.HeatMap(
        heat_data, 
        name="Heat Map",
        gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
    )
    heat_map.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add clustering statistics with bright styling
    cluster_stats = {}
    for label in unique_labels:
        count = np.sum(labels == label)
        if label == -1:
            cluster_stats["Noise Points"] = count
        else:
            cluster_stats[f"Cluster {label}"] = count
    
    # Create statistics HTML with bright background
    stats_html = f"""
    <div style='position: fixed; top: 10px; right: 10px; width: 250px; height: auto; 
                background-color: rgba(255,255,255,0.95); border: 2px solid #333; 
                border-radius: 8px; z-index: 9999; font-size: 14px; padding: 15px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
        <h4 style='margin-top: 0; color: #333;'>Truck Trajectory Clustering</h4>
        <p><b>Data Source:</b> {data_info['source']}</p>
        <p><b>Date:</b> {', '.join(data_info['dates'])}</p>
        <p><b>Total Points:</b> {len(data)}</p>
        <p><b>Clusters Found:</b> {len(unique_labels) - (1 if -1 in unique_labels else 0)}</p>
    """
    
    for cluster_name, count in cluster_stats.items():
        stats_html += f"<p><b>{cluster_name}:</b> {count} points</p>"
    
    stats_html += "</div>"
    
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # Save map to output directory
    full_path = os.path.join(output_dir, save_path)
    m.save(full_path)
    print(f"Interactive map saved as: {full_path}")
    
    return m

def plot_clustering_analysis(data, labels, data_info, output_dir, save_plots=True):
    """Create comprehensive clustering analysis plots with fixed overlapping issues"""
    
    # Create figure with better layout and spacing
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3, 
                         left=0.08, right=0.95, top=0.93, bottom=0.07)
    
    fig.suptitle(f'DBSCAN Clustering Analysis - Truck Trajectory Data ({data_info["source"]})', 
                 fontsize=16, fontweight='bold', y=0.97)
    
    # Define consistent colors
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # 1. Geographic scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    for i, label in enumerate(unique_labels):
        if label == -1:
            cluster_data = data[labels == label]
            ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                       c='black', marker='x', s=50, alpha=0.8, label='Noise')
        else:
            cluster_data = data[labels == label]
            ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                       c=[colors[i]], s=60, alpha=0.7, label=f'Cluster {label}')
    
    ax1.set_title('Geographic Distribution of Clusters', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude (°E)')
    ax1.set_ylabel('Latitude (°N)')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cluster size distribution
    ax2 = fig.add_subplot(gs[0, 1])
    cluster_sizes = []
    cluster_names = []
    
    for label in unique_labels:
        size = np.sum(labels == label)
        if label == -1:
            cluster_names.append('Noise')
        else:
            cluster_names.append(f'Cluster {label}')
        cluster_sizes.append(size)
    
    bars = ax2.bar(range(len(cluster_sizes)), cluster_sizes, 
                   color=colors[:len(cluster_sizes)])
    ax2.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Clusters')
    ax2.set_ylabel('Number of Points')
    ax2.set_xticks(range(len(cluster_names)))
    ax2.set_xticklabels(cluster_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, size in zip(bars, cluster_sizes):
        ax2.annotate(f'{size}', (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5),
                    ha='center', va='bottom', fontsize=8)
    
    # 3. Distance distribution
    ax3 = fig.add_subplot(gs[0, 2])
    from scipy.spatial.distance import pdist
    distances = pdist(data)
    ax3.hist(distances, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax3.set_title('Pairwise Distance Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Distance (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.mean(distances), color='red', linestyle='--', 
                label=f'Mean: {np.mean(distances):.4f}')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Coordinate distributions
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(data[:, 0], bins=30, alpha=0.7, label='Longitude', color='blue', edgecolor='black')
    ax4.set_title('Longitude Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Longitude (°E)')
    ax4.set_ylabel('Frequency')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Latitude distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(data[:, 1], bins=30, alpha=0.7, label='Latitude', color='red', edgecolor='black')
    ax5.set_title('Latitude Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Latitude (°N)')
    ax5.set_ylabel('Frequency')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Cluster compactness
    ax6 = fig.add_subplot(gs[1, 2])
    cluster_compactness = []
    valid_clusters = [label for label in unique_labels if label != -1]
    
    for label in valid_clusters:
        cluster_data = data[labels == label]
        if len(cluster_data) > 1:
            cluster_distances = pdist(cluster_data)
            cluster_compactness.append(np.mean(cluster_distances))
        else:
            cluster_compactness.append(0)
    
    if cluster_compactness:
        bars = ax6.bar(range(len(valid_clusters)), cluster_compactness, 
                       color=colors[:len(valid_clusters)])
        ax6.set_title('Cluster Compactness\n(Avg Intra-cluster Distance)', 
                      fontsize=12, fontweight='bold')
        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Average Distance (degrees)')
        ax6.set_xticks(range(len(valid_clusters)))
        ax6.set_xticklabels([f'C{label}' for label in valid_clusters])
        
        # Add value labels
        for bar, comp in zip(bars, cluster_compactness):
            ax6.annotate(f'{comp:.4f}', (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001),
                        ha='center', va='bottom', fontsize=8)
    
    # 7. Geographic bounds visualization
    ax7 = fig.add_subplot(gs[2, :2])
    
    # Calculate convex hulls for each cluster
    from scipy.spatial import ConvexHull
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
            
        cluster_data = data[labels == label]
        if len(cluster_data) >= 3:  # Need at least 3 points for convex hull
            try:
                hull = ConvexHull(cluster_data)
                for simplex in hull.simplices:
                    ax7.plot(cluster_data[simplex, 0], cluster_data[simplex, 1], 
                            color=colors[i], alpha=0.7, linewidth=2)
                ax7.fill(cluster_data[hull.vertices, 0], cluster_data[hull.vertices, 1], 
                        color=colors[i], alpha=0.3)
            except:
                pass
        
        # Plot points
        ax7.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                   c=[colors[i]], s=40, alpha=0.8, label=f'Cluster {label}')
    
    # Plot noise points
    if -1 in unique_labels:
        noise_data = data[labels == -1]
        ax7.scatter(noise_data[:, 0], noise_data[:, 1], 
                   c='black', marker='x', s=60, alpha=0.8, label='Noise')
    
    ax7.set_title('Cluster Boundaries (Convex Hulls)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Longitude (°E)')
    ax7.set_ylabel('Latitude (°N)')
    ax7.legend(fontsize=8, loc='best')
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Create summary text
    summary_text = "CLUSTERING SUMMARY\n" + "="*20 + "\n\n"
    summary_text += f"Data Source: {data_info['source']}\n"
    summary_text += f"Date Range: {', '.join(data_info['dates'])}\n"
    summary_text += f"Total Points: {len(data)}\n"
    summary_text += f"Clusters Found: {len(unique_labels) - (1 if -1 in unique_labels else 0)}\n"
    summary_text += f"Noise Points: {np.sum(labels == -1)}\n\n"
    
    summary_text += "GEOGRAPHIC BOUNDS:\n" + "-"*15 + "\n"
    summary_text += f"Lon: [{data_info['longitude_range'][0]:.4f}, {data_info['longitude_range'][1]:.4f}]\n"
    summary_text += f"Lat: [{data_info['latitude_range'][0]:.4f}, {data_info['latitude_range'][1]:.4f}]\n\n"
    
    summary_text += "CLUSTER DETAILS:\n" + "-"*15 + "\n"
    for label in sorted(unique_labels):
        count = np.sum(labels == label)
        percentage = (count / len(data)) * 100
        if label == -1:
            summary_text += f"Noise: {count} pts ({percentage:.1f}%)\n"
        else:
            summary_text += f"Cluster {label}: {count} pts ({percentage:.1f}%)\n"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    if save_plots:
        plot_path = os.path.join(output_dir, 'truck_trajectory_clustering_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Analysis plots saved as: {plot_path}")
    
    plt.show()

def main():
    
    """Enhanced main function with real truck trajectory data analysis"""
    
    print("=== Enhanced DBSCAN Clustering with Real Truck Trajectory Data ===\n")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory created: {output_dir}\n")
    
    # Load real truck trajectory data
    print("Loading truck trajectory data...")
    base_path = "E:/dwd_truck"
    dates = ["2024-04-18"]
    
    # Load data with optional sampling to manage large datasets
    data, data_info = load_truck_trajectory_data(
        base_path=base_path, 
        dates=dates, 
        n_samples=None  # Limit to 1000 points for demo, set to None for all data
    )
    
    print(f"\nDataset Information:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")
    
    # Initialize dynamic DBSCAN
    print("\nInitializing Dynamic DBSCAN...")
    
    # Parameters optimized for geographic coordinates
    # You may need to adjust these based on your data characteristics
    '''
    eps = 0.01  # Will be scaled down in mock implementation
    min_samples = 5
    '''
    alpha = 0.1
    beta = 2
    
    eps, min_samples, Ada_t = enhanced_adaptive_dbscan_params(data.shape[0], beta,data, use_optimization=True, verbose=True)
    print(f"Using parameters: eps={eps}, min_samples={min_samples}, alpha={alpha}, beta={beta}\n")

    # Determine initial subset size
    initial_size = min(100, len(data) // 2)
    
    # Initialize with first subset of points
    dbscan_alg = MockDynamicDBSCAN(eps, min_samples, alpha, beta, data[:initial_size, :])
    
    # Get initial clustering
    predicted_labels_initial = [dbscan_alg.get_cluster(i) for i in range(initial_size)]
    print(f"Initial clustering ({initial_size} points): {len(set(predicted_labels_initial))} clusters")
    
    # Add remaining points incrementally
    if len(data) > initial_size:
        print("Adding remaining points incrementally...")
        for i in range(initial_size, len(data)):
            dbscan_alg.add_point(data[i, :])
            if (i + 1) % 100 == 0:
                temp_labels = [dbscan_alg.get_cluster(j) for j in range(i + 1)]
                print(f"After {i + 1} points: {len(set(temp_labels))} clusters")
    
    # Get final clustering results
    predicted_labels = [dbscan_alg.get_cluster(i) for i in range(len(data))]
    predicted_labels = np.array(predicted_labels)
    
    # Print detailed statistics
    print("\n=== Final Clustering Results ===")
    unique_clusters = set(predicted_labels)
    print(f"Total number of clusters: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
    print(f"Cluster labels: {sorted(unique_clusters)}")
    
    for cluster_id in sorted(unique_clusters):
        count = np.sum(predicted_labels == cluster_id)
        percentage = (count / len(data)) * 100
        if cluster_id == -1:
            print(f"Noise points: {count} ({percentage:.1f}%)")
        else:
            print(f"Cluster {cluster_id}: {count} points ({percentage:.1f}%)")
    
    # Calculate clustering quality metrics
    if len(set(predicted_labels)) > 1 and -1 not in predicted_labels:
        try:
            silhouette_avg = silhouette_score(data, predicted_labels)
            print(f"\nSilhouette Score: {silhouette_avg:.3f}")
        except:
            print("\nSilhouette Score: Could not calculate (noise points present)")
    
    # Create comprehensive analysis plots
    print("\nGenerating analysis plots...")
    plot_clustering_analysis(data, predicted_labels, data_info, output_dir, save_plots=True)
    
    # Create interactive map
    print("Creating interactive map...")
    interactive_map = create_interactive_map(data, predicted_labels, data_info, output_dir)
    
    # Export results to CSV
    results_df = pd.DataFrame({
        'longitude': data[:, 0],
        'latitude': data[:, 1],
        'cluster': predicted_labels,
        'point_index': range(len(data))
    })
    
    csv_filename = f"truck_trajectory_clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"Results exported to: {csv_path}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Data source: {data_info['source']}")
    print(f"Date range: {', '.join(data_info['dates'])}")
    print(f"Total data points processed: {len(data)}")
    print(f"Final number of clusters: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
    print(f"Noise points: {np.sum(predicted_labels == -1)}")
    print(f"Geographic coverage: Lon[{data_info['longitude_range'][0]:.4f}, {data_info['longitude_range'][1]:.4f}], Lat[{data_info['latitude_range'][0]:.4f}, {data_info['latitude_range'][1]:.4f}]")
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - Interactive map: truck_dbscan_clusters.html")
    print(f"  - Analysis plots: truck_trajectory_clustering_analysis.png")
    print(f"  - Results data: {csv_filename}")
    
    return data, predicted_labels, interactive_map, data_info

if __name__ == "__main__":
    main()