import pandas as pd
import os
import math

def convert_gps_to_coordinates(longitude, latitude, scale_factor=100000):
    """
    将GPS坐标转换为整数坐标
    
    Args:
        longitude: 经度
        latitude: 纬度
        scale_factor: 缩放因子，用于将小数坐标转换为整数
    
    Returns:
        tuple: (x, y) 整数坐标
    """
    x = int(longitude * scale_factor)
    y = int(latitude * scale_factor)
    return x, y

def is_in_chengdu(longitude, latitude):
    """
    判断坐标是否在成都市范围内
    
    Args:
        longitude: 经度
        latitude: 纬度
    
    Returns:
        bool: True表示在成都市范围内，False表示不在
    """
    # 成都市地理边界：东经102°54′~104°53′、北纬30°05′~31°26′
    # 转换为十进制度数
    min_longitude = 102.9  # 102°54′
    max_longitude = 104.53  # 104°53′
    min_latitude = 30.08   # 30°05′
    max_latitude = 31.26   # 31°26′
    
    return (min_longitude <= longitude <= max_longitude and 
            min_latitude <= latitude <= max_latitude)

def create_vrp_file(csv_file_path, output_file_path=None, min_count=10, 
                   capacity=999999, depot_demand=0, customer_demand=1):
    """
    将合并后的聚类数据转换为VRP文件格式，同时生成节点信息CSV文件
    
    Args:
        csv_file_path: 输入的CSV文件路径
        output_file_path: 输出的VRP文件路径，如果为None则自动生成
        min_count: 最小点数阈值，只处理count >= min_count的行
        capacity: 车辆容量（设置为很大的值）
        depot_demand: 仓库需求量（通常为0）
        customer_demand: 客户需求量（设置为很小的值）
    """
    try:
        # 读取CSV文件
        if not os.path.exists(csv_file_path):
            print(f"错误: 文件 '{csv_file_path}' 不存在。")
            return None, None
        
        df = pd.read_csv(csv_file_path)
        print(f"成功读取文件: {csv_file_path}")
        print(f"原始数据行数: {len(df)}")
        
        # 检查必需的列
        required_columns = ['cluster', 'longitude', 'latitude', 'count']
        if not all(col in df.columns for col in required_columns):
            print(f"错误: CSV文件缺少必要的列。需要: {required_columns}")
            print(f"实际列: {list(df.columns)}")
            return None, None
        
        # 筛选count >= min_count的行
        df_filtered = df[df['count'] >= min_count].copy()
        print(f"筛选后数据行数 (count >= {min_count}): {len(df_filtered)}")
        
        # 筛选成都市范围内的坐标
        chengdu_mask = df_filtered.apply(lambda row: is_in_chengdu(row['longitude'], row['latitude']), axis=1)
        df_chengdu = df_filtered[chengdu_mask].copy()
        
        excluded_count = len(df_filtered) - len(df_chengdu)
        print(f"剔除成都市以外的坐标点数: {excluded_count}")
        print(f"成都市内有效数据行数: {len(df_chengdu)}")
        
        if df_chengdu.empty:
            print(f"没有找到成都市内且count >= {min_count}的数据行。")
            return None, None
        
        # 显示筛选后的数据
        print("\n成都市内筛选后的聚类数据:")
        for index, row in df_chengdu.iterrows():
            print(f"  聚类 {int(row['cluster'])}: 经度 {row['longitude']:.6f}, "
                  f"纬度 {row['latitude']:.6f}, 点数 {int(row['count'])}")
        
        # 如果有被排除的点，显示信息
        if excluded_count > 0:
            print(f"\n被排除的成都市以外的坐标点:")
            excluded_df = df_filtered[~chengdu_mask]
            for index, row in excluded_df.iterrows():
                print(f"  聚类 {int(row['cluster'])}: 经度 {row['longitude']:.6f}, "
                      f"纬度 {row['latitude']:.6f}, 点数 {int(row['count'])} (成都市外)")
        
        # 重新排序并分配节点编号（从2开始，1为仓库）
        df_chengdu = df_chengdu.reset_index(drop=True)
        df_chengdu['node_id'] = range(2, len(df_chengdu) + 2)  # 从2开始编号
        
        # 转换GPS坐标为整数坐标
        coordinates = []
        for index, row in df_chengdu.iterrows():
            x, y = convert_gps_to_coordinates(row['longitude'], row['latitude'])
            coordinates.append((int(row['node_id']), x, y))  # 确保node_id是整数
        
        # 设置仓库节点坐标为成都市交管局坐标
        # 成都市交管局大概位置：成都市中心区域，使用成都市政府坐标作为参考
        depot_longitude = 104.07  # 成都市中心经度
        depot_latitude = 30.67    # 成都市中心纬度
        depot_x, depot_y = convert_gps_to_coordinates(depot_longitude, depot_latitude)
        
        print(f"\n仓库设置信息:")
        print(f"  仓库坐标(GPS): 经度 {depot_longitude:.6f}, 纬度 {depot_latitude:.6f}")
        print(f"  仓库坐标(转换后): ({depot_x}, {depot_y})")
        
        # 计算维度（包括仓库）
        dimension = len(df_chengdu) + 1
        
        # 生成输出文件名
        if output_file_path is None:
            base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
            output_file_path = f"{base_name}_chengdu_min{min_count}.vrp"
        
        # 生成节点信息CSV文件路径
        csv_output_path = os.path.splitext(output_file_path)[0] + "_nodes.csv"
        
        # 创建节点信息DataFrame
        nodes_data = []
        
        # 添加仓库节点信息
        nodes_data.append({
            'node_id': 1,
            'node_type': 'depot',
            'longitude': depot_longitude,
            'latitude': depot_latitude,
            'x_coord': depot_x,
            'y_coord': depot_y,
            'demand': depot_demand,
            'original_cluster': 'depot',
            'count': 0,
            'description': '成都市交管局(仓库)'
        })
        
        # 添加客户节点信息
        for index, row in df_chengdu.iterrows():
            x, y = convert_gps_to_coordinates(row['longitude'], row['latitude'])
            nodes_data.append({
                'node_id': int(row['node_id']),
                'node_type': 'customer',
                'longitude': row['longitude'],
                'latitude': row['latitude'],
                'x_coord': x,
                'y_coord': y,
                'demand': customer_demand,
                'original_cluster': int(row['cluster']),
                'count': int(row['count']),
                'description': f'客户节点(原聚类{int(row["cluster"])})'
            })
        
        # 创建节点信息DataFrame并保存为CSV
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(csv_output_path, index=False, encoding='utf-8')
        
        print(f"\n成功生成节点信息CSV文件: {csv_output_path}")
        
        # 写入VRP文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # 文件头信息
            f.write(f"NAME : \t{os.path.splitext(os.path.basename(output_file_path))[0]}\n")
            f.write(f"COMMENT : \t\"Generated from Chengdu merged cluster data with count >= {min_count}\"\n")
            f.write(f"TYPE : \tCVRP\n")
            f.write(f"DIMENSION : \t{dimension}\n")
            f.write(f"EDGE_WEIGHT_TYPE : \tEUC_2D\n")
            f.write(f"CAPACITY : \t{capacity}\n")
            
            # 节点坐标部分
            f.write(f"NODE_COORD_SECTION\n")
            # 仓库节点（编号1）- 成都市交管局坐标
            f.write(f"1\t{depot_x}\t{depot_y}\n")
            # 客户节点 - 使用整数格式
            for node_id, x, y in coordinates:
                f.write(f"{node_id}\t{x}\t{y}\n")  # node_id已经是整数
            
            # 需求部分
            f.write(f"DEMAND_SECTION\n")
            # 仓库需求
            f.write(f"1\t{depot_demand}\n")
            # 客户需求 - 使用整数格式
            for index, row in df_chengdu.iterrows():
                f.write(f"{int(row['node_id'])}\t{customer_demand}\n")  # 确保是整数
            
            # 仓库部分
            f.write(f"DEPOT_SECTION\n")
            f.write(f"\t1\n")
            f.write(f"\t-1\n")
            f.write(f"EOF\n")
        
        print(f"\n成功生成VRP文件: {output_file_path}")
        print(f"VRP问题规模:")
        print(f"  - 总节点数: {dimension} (包括1个仓库)")
        print(f"  - 客户节点数: {len(df_chengdu)} (仅成都市内)")
        print(f"  - 车辆容量: {capacity}")
        print(f"  - 客户需求: {customer_demand}")
        print(f"  - 仓库坐标: ({depot_x}, {depot_y}) [成都市交管局]")
        
        # 显示节点信息摘要
        print(f"\n节点信息摘要:")
        print(f"  节点1 (仓库): 坐标 ({depot_x}, {depot_y}), 需求 {depot_demand} [成都市交管局]")
        for index, row in df_chengdu.iterrows():
            x, y = convert_gps_to_coordinates(row['longitude'], row['latitude'])
            print(f"  节点{int(row['node_id'])} (客户): 坐标 ({x}, {y}), 需求 {customer_demand}, 原始点数 {int(row['count'])}")
        
        # 显示CSV文件内容预览
        print(f"\n节点信息CSV文件内容预览:")
        print(nodes_df.to_string(index=False, max_rows=10))
        if len(nodes_df) > 10:
            print(f"... (共 {len(nodes_df)} 行)")
        
        # 验证生成的VRP文件格式
        print(f"\n验证VRP文件格式...")
        verify_vrp_file_format(output_file_path)
        
        return output_file_path, csv_output_path
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        return None, None

def verify_vrp_file_format(vrp_file_path):
    """
    验证VRP文件格式，特别是节点ID是否为整数格式
    
    Args:
        vrp_file_path: VRP文件路径
    """
    try:
        with open(vrp_file_path, 'r') as f:
            lines = f.readlines()
        
        in_node_section = False
        in_demand_section = False
        node_errors = []
        demand_errors = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            if line == "NODE_COORD_SECTION":
                in_node_section = True
                in_demand_section = False
                continue
            elif line == "DEMAND_SECTION":
                in_node_section = False
                in_demand_section = True
                continue
            elif line in ["DEPOT_SECTION", "EOF"] or line.startswith(("NAME", "COMMENT", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE", "CAPACITY")):
                in_node_section = False
                in_demand_section = False
                continue
            
            # 检查节点坐标部分
            if in_node_section and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        node_id = int(parts[0])  # 尝试转换为整数
                    except ValueError:
                        node_errors.append(f"行 {line_num}: 节点ID '{parts[0]}' 不是整数格式")
            
            # 检查需求部分
            if in_demand_section and line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        node_id = int(parts[0])  # 尝试转换为整数
                    except ValueError:
                        demand_errors.append(f"行 {line_num}: 节点ID '{parts[0]}' 不是整数格式")
        
        if node_errors or demand_errors:
            print("发现格式问题:")
            for error in node_errors + demand_errors:
                print(f"  {error}")
            return False
        else:
            print("VRP文件格式验证通过 - 所有节点ID都是整数格式")
            return True
            
    except Exception as e:
        print(f"验证VRP文件格式时出错: {e}")
        return False

def generate_multiple_vrp_files(csv_file_path, output_dir, min_count_list):
    """
    根据不同的min_count值生成多个VRP文件和对应的节点信息CSV文件
    
    Args:
        csv_file_path: 输入的CSV文件路径
        output_dir: 输出目录
        min_count_list: min_count值的列表
    
    Returns:
        list: 生成的文件路径列表（包含VRP文件和CSV文件）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    print("=" * 80)
    print(f"开始批量生成VRP文件和节点信息CSV文件，输出目录: {output_dir}")
    print(f"min_count值列表: {min_count_list}")
    print("=" * 80)
    
    for i, min_count in enumerate(min_count_list, 1):
        print(f"\n{'=' * 60}")
        print(f"第 {i}/{len(min_count_list)} 步：生成 min_count={min_count} 的文件")
        print("=" * 60)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        output_file_name = f"{base_name}_chengdu_min{min_count}.vrp"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        # 生成VRP文件和节点信息CSV文件
        vrp_file, csv_file = create_vrp_file(csv_file_path, output_file_path, min_count)
        
        if vrp_file and csv_file:
            generated_files.extend([vrp_file, csv_file])
            print(f"✓ 成功生成VRP文件: {vrp_file}")
            print(f"✓ 成功生成节点CSV文件: {csv_file}")
        else:
            print(f"✗ 生成文件失败: min_count={min_count}")
    
    print("\n" + "=" * 80)
    print("批量生成完成总结:")
    print("=" * 80)
    print(f"总共尝试生成: {len(min_count_list) * 2} 个文件 ({len(min_count_list)} VRP + {len(min_count_list)} CSV)")
    print(f"成功生成: {len(generated_files)} 个文件")
    print(f"失败: {len(min_count_list) * 2 - len(generated_files)} 个文件")
    
    if generated_files:
        print("\n成功生成的文件列表:")
        vrp_files = [f for f in generated_files if f.endswith('.vrp')]
        csv_files = [f for f in generated_files if f.endswith('.csv')]
        
        print(f"  VRP文件 ({len(vrp_files)} 个):")
        for file_path in vrp_files:
            file_size = os.path.getsize(file_path)
            print(f"    - {file_path} ({file_size} bytes)")
            
        print(f"  节点信息CSV文件 ({len(csv_files)} 个):")
        for file_path in csv_files:
            file_size = os.path.getsize(file_path)
            print(f"    - {file_path} ({file_size} bytes)")
    
    return generated_files

def analyze_coordinate_distribution(csv_file_path):
    """
    分析坐标分布，显示成都市内外的数据统计
    
    Args:
        csv_file_path: CSV文件路径
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"坐标分布分析:")
        print(f"总数据行数: {len(df)}")
        
        # 统计成都市内外的数据
        chengdu_mask = df.apply(lambda row: is_in_chengdu(row['longitude'], row['latitude']), axis=1)
        chengdu_count = chengdu_mask.sum()
        outside_count = len(df) - chengdu_count
        
        print(f"成都市内坐标: {chengdu_count} 行")
        print(f"成都市外坐标: {outside_count} 行")
        
        # 显示经纬度范围
        print(f"\n坐标范围统计:")
        print(f"经度范围: {df['longitude'].min():.6f} ~ {df['longitude'].max():.6f}")
        print(f"纬度范围: {df['latitude'].min():.6f} ~ {df['latitude'].max():.6f}")
        
        # 成都市内数据的范围
        if chengdu_count > 0:
            chengdu_df = df[chengdu_mask]
            print(f"\n成都市内坐标范围:")
            print(f"经度范围: {chengdu_df['longitude'].min():.6f} ~ {chengdu_df['longitude'].max():.6f}")
            print(f"纬度范围: {chengdu_df['latitude'].min():.6f} ~ {chengdu_df['latitude'].max():.6f}")
            
            # 分析不同count阈值下的数据量
            print(f"\n不同count阈值下的成都市内数据量:")
            for threshold in [10, 20, 30, 50, 100, 200, 500]:
                count = len(chengdu_df[chengdu_df['count'] >= threshold])
                print(f"  count >= {threshold:3d}: {count:3d} 行")
        
    except Exception as e:
        print(f"分析坐标分布时出错: {e}")

# 运行脚本
if __name__ == "__main__":
    
    csv_path = 'cluster_res/all_merged_clusters.csv'
    output_directory = r'elg-master\cvrp\truck_data'  # 目标输出目录
    
    # 定义不同的min_count值列表
    min_count_values = [100]
    
    # 分析坐标分布
    print("=" * 80)
    print("第一步：分析坐标分布")
    print("=" * 80)
    analyze_coordinate_distribution(csv_path)
    
    print("\n" + "=" * 80)
    print("第二步：批量生成不同min_count的VRP文件和节点信息CSV文件")
    print("=" * 80)
    
    # 批量生成VRP文件和节点信息CSV文件
    generated_files = generate_multiple_vrp_files(csv_path, output_directory, min_count_values)
    if generated_files:
        print(f"\n所有文件已保存到目录: {output_directory}")
    else:
        print(f"\n未生成任何文件，请检查输入数据和参数设置。")