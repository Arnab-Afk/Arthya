import React, { useState, useEffect, useCallback } from 'react';
import { StyleSheet, View, Text, ScrollView, TouchableOpacity, ActivityIndicator, RefreshControl } from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { Svg, Rect, Defs, LinearGradient as SvgLinearGradient, Stop, Text as SvgText, Circle, Line } from 'react-native-svg';
import { Dimensions } from 'react-native';
import Animated, { FadeInDown, FadeInUp, FadeIn } from 'react-native-reanimated';
import { useRouter } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import api from '@/services/api';
import { CategorySpending } from '@/types/api';

const { width } = Dimensions.get('window');

type TimePeriod = 'weekly' | 'monthly' | 'annually';

interface CategoryData {
    name: string;
    amount: number;
    percentage: number;
    color: string;
    icon: keyof typeof Ionicons.glyphMap;
    trend: number;
}

const categoryColors = ['#9FE8AE', '#D4B483', '#FF9500', '#64D2FF', '#FF453A', '#BF5AF2'];

const getCategoryIcon = (category: string): keyof typeof Ionicons.glyphMap => {
    const icons: Record<string, keyof typeof Ionicons.glyphMap> = {
        'food': 'restaurant',
        'food_dining': 'restaurant',
        'shopping': 'cart',
        'transport': 'car',
        'transportation': 'car',
        'entertainment': 'game-controller',
        'utilities': 'flash',
        'healthcare': 'medkit',
        'health': 'medkit',
        'bills': 'receipt',
        'income': 'wallet',
        'other': 'ellipsis-horizontal',
    };
    return icons[category.toLowerCase()] || 'ellipsis-horizontal';
};

export default function ChartsScreen() {
    const router = useRouter();
    const [selectedPeriod, setSelectedPeriod] = useState<TimePeriod>('monthly');
    const [selectedCategory, setSelectedCategory] = useState<number | null>(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [spending, setSpending] = useState(0);
    const [income, setIncome] = useState(0);
    const [data, setData] = useState<CategoryData[]>([]);
    const [isInitialLoad, setIsInitialLoad] = useState(true);

    // Refresh data when screen comes into focus OR when period changes
    useFocusEffect(
        useCallback(() => {
            loadData();
        }, [selectedPeriod])
    );

    const loadData = async () => {
        try {
            // Only show loading spinner on initial load
            if (isInitialLoad) {
                setLoading(true);
            }
            
            // Get days based on period - fetch real data for each period
            const days = selectedPeriod === 'weekly' ? 7 : selectedPeriod === 'monthly' ? 30 : 365;
            
            const response = await api.getSpendingAnalysis(days);
            
            if (response.success && response.data) {
                const analyticsData = response.data;
                const totalSpending = analyticsData.totalSpending || analyticsData.totalExpense || 0;
                const totalIncome = analyticsData.totalIncome || 0;
                
                setSpending(totalSpending);
                setIncome(totalIncome);
                
                // Transform category breakdown into chart data
                const categories = analyticsData.categories || analyticsData.categoryBreakdown || [];
                
                if (categories.length > 0) {
                    const categoryData: CategoryData[] = categories.map((cat: CategorySpending, index: number) => {
                        const amount = typeof cat.total === 'string' ? parseFloat(cat.total) : cat.total || 0;
                        const percentage = typeof cat.percentage === 'string' ? parseFloat(cat.percentage) : cat.percentage || 0;
                        
                        return {
                            name: cat.category.charAt(0).toUpperCase() + cat.category.slice(1).replace('_', ' '),
                            amount,
                            percentage,
                            color: categoryColors[index % categoryColors.length],
                            icon: getCategoryIcon(cat.category),
                            trend: 0,
                        };
                    });
                    
                    setData(categoryData);
                } else {
                    setData(getEmptyData());
                }
            } else {
                setData(getEmptyData());
                setSpending(0);
                setIncome(0);
            }
        } catch (error) {
            console.error('Failed to load chart data:', error);
            setData(getEmptyData());
            setSpending(0);
            setIncome(0);
        } finally {
            if (isInitialLoad) {
                setLoading(false);
                setIsInitialLoad(false);
            }
        }
    };

    const getEmptyData = (): CategoryData[] => [
        { name: 'No Data', amount: 0, percentage: 100, color: Colors.textDim, icon: 'help-circle', trend: 0 }
    ];

    const onRefresh = async () => {
        setRefreshing(true);
        await loadData();
        setRefreshing(false);
    };

    const formatCurrency = (amount: number) => {
        return `₹${amount.toLocaleString('en-IN', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
    };

    if (loading) {
        return (
            <View style={[styles.container, styles.centerContent]}>
                <ActivityIndicator size="large" color={Colors.primary} />
                <Text style={styles.loadingText}>Loading charts...</Text>
            </View>
        );
    }

    const maxAmount = Math.max(...data.map(d => d.amount), 1);
    const chartHeight = 220;
    const chartWidth = width - 80;
    const barWidth = Math.min(40, chartWidth / Math.max(data.length, 1) - 12);
    const barSpacing = (chartWidth - (barWidth * data.length)) / (data.length + 1);
    const borderRadius = 8;
    const displaySpending = spending || data.reduce((sum, d) => sum + d.amount, 0);

    return (
        <View style={styles.container}>
            <Animated.View entering={FadeInDown.delay(100).duration(500)} style={styles.header}>
                <View style={styles.headerLeft}>
                    <Ionicons name="stats-chart" size={28} color={Colors.primary} />
                    <Text style={styles.headerTitle}>Spending Analysis</Text>
                </View>
                <View style={styles.headerRight}>
                    <TouchableOpacity style={styles.filterButton} onPress={onRefresh}>
                        <Ionicons name="refresh" size={20} color={Colors.text} />
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.profileButton} onPress={() => router.push('/explore')}>
                        <Ionicons name="person-circle-outline" size={28} color={Colors.text} />
                    </TouchableOpacity>
                </View>
            </Animated.View>

            <Animated.View entering={FadeInDown.delay(200).duration(500)} style={styles.tabsContainer}>
                <TouchableOpacity 
                    style={[styles.tab, selectedPeriod === 'weekly' && styles.activeTab]}
                    onPress={() => setSelectedPeriod('weekly')}
                >
                    <Text style={selectedPeriod === 'weekly' ? styles.activeTabText : styles.tabText}>Weekly</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                    style={[styles.tab, selectedPeriod === 'monthly' && styles.activeTab]}
                    onPress={() => setSelectedPeriod('monthly')}
                >
                    <Text style={selectedPeriod === 'monthly' ? styles.activeTabText : styles.tabText}>Monthly</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                    style={[styles.tab, selectedPeriod === 'annually' && styles.activeTab]}
                    onPress={() => setSelectedPeriod('annually')}
                >
                    <Text style={selectedPeriod === 'annually' ? styles.activeTabText : styles.tabText}>Annually</Text>
                </TouchableOpacity>
            </Animated.View>

            <ScrollView 
                contentContainerStyle={styles.content} 
                showsVerticalScrollIndicator={false}
                refreshControl={
                    <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={Colors.primary} />
                }
            >
                {/* Total Spending Card */}
                <Animated.View entering={FadeInUp.delay(300).duration(600)} style={styles.totalCard}>
                    <View style={styles.totalCardHeader}>
                        <View>
                            <Text style={styles.totalLabel}>Total Spending</Text>
                            <Text style={styles.totalAmount}>{formatCurrency(displaySpending)}</Text>
                        </View>
                        {income > 0 && (
                            <View style={styles.trendBadge}>
                                <Ionicons 
                                    name={displaySpending < income ? "trending-down" : "trending-up"} 
                                    size={16} 
                                    color={displaySpending < income ? Colors.success : Colors.danger} 
                                />
                                <Text style={[
                                    styles.trendText,
                                    { color: displaySpending < income ? Colors.success : Colors.danger }
                                ]}>
                                    {((displaySpending / income) * 100).toFixed(0)}% of income
                                </Text>
                            </View>
                        )}
                    </View>
                    <Text style={styles.comparisonText}>
                        {selectedPeriod === 'weekly' ? 'This week' : selectedPeriod === 'monthly' ? 'This month' : 'This year'}
                    </Text>
                </Animated.View>

                {/* Interactive Chart */}
                {data.length > 0 && data[0].amount > 0 ? (
                    <Animated.View entering={FadeInUp.delay(500).duration(800)} style={styles.chartContainer}>
                        <Svg height={chartHeight + 60} width={chartWidth}>
                            <Defs>
                                {data.map((category, index) => (
                                    <SvgLinearGradient key={index} id={`grad${index}`} x1="0" y1="0" x2="0" y2="1">
                                        <Stop offset="0" stopColor={category.color} stopOpacity="1" />
                                        <Stop offset="1" stopColor={category.color} stopOpacity="0.5" />
                                    </SvgLinearGradient>
                                ))}
                            </Defs>

                            {/* Grid lines */}
                            {[0, 25, 50, 75, 100].map((percent, i) => {
                                const y = chartHeight - (chartHeight * percent / 100);
                                return (
                                    <Line
                                        key={i}
                                        x1="0"
                                        y1={y}
                                        x2={chartWidth}
                                        y2={y}
                                        stroke={Colors.textDim}
                                        strokeWidth="1"
                                        strokeOpacity={0.1}
                                        strokeDasharray="8,8"
                                    />
                                );
                            })}

                            {/* Bars */}
                            {data.map((category, index) => {
                                // Reduced max height to prevent overflow and leave space for labels
                                const barHeight = Math.max((category.amount / maxAmount) * (chartHeight - 60), 10);
                                const x = barSpacing + (index * (barWidth + barSpacing));
                                const y = chartHeight - barHeight;
                                const isSelected = index === selectedCategory;
                                const isFocused = selectedCategory === null || isSelected;

                                return (
                                    <React.Fragment key={index}>
                                        {/* Background bar */}
                                        <Rect
                                            x={x}
                                            y={20}
                                            width={barWidth}
                                            height={chartHeight - 20}
                                            rx={borderRadius}
                                            fill="rgba(255, 255, 255, 0.03)"
                                        />
                                        {/* Value bar */}
                                        <Rect
                                            x={x}
                                            y={y}
                                            width={barWidth}
                                            height={barHeight}
                                            rx={borderRadius}
                                            fill={`url(#grad${index})`}
                                            opacity={isFocused ? 1 : 0.3}
                                        />
                                        {/* Invisible touch target - full height for better clickability */}
                                        <Rect
                                            x={x - 5}
                                            y={0}
                                            width={barWidth + 10}
                                            height={chartHeight + 40}
                                            fill="transparent"
                                            onPress={() => setSelectedCategory(index === selectedCategory ? null : index)}
                                        />
                                        {/* Info on click */}
                                        {isSelected && (
                                            <SvgText
                                                x={x + barWidth / 2}
                                                y={y - 12}
                                                fill={Colors.text}
                                                fontSize="14"
                                                fontWeight="bold"
                                                textAnchor="middle"
                                            >
                                                {formatCurrency(category.amount)}
                                            </SvgText>
                                        )}
                                        {/* Labels */}
                                        <SvgText
                                            x={x + barWidth / 2}
                                            y={chartHeight + 25}
                                            fill={isSelected ? Colors.text : Colors.textDim}
                                            fontSize="11"
                                            fontWeight={isSelected ? '600' : '400'}
                                            textAnchor="middle"
                                            opacity={isFocused ? 1 : 0.5}
                                        >
                                            {category.name.length > 8 ? category.name.substring(0, 6) + '..' : category.name}
                                        </SvgText>
                                    </React.Fragment>
                                );
                            })}
                        </Svg>
                    </Animated.View>
                ) : (
                    <Animated.View entering={FadeInUp.delay(500).duration(800)} style={styles.emptyChart}>
                        <Ionicons name="bar-chart-outline" size={64} color={Colors.textDim} />
                        <Text style={styles.emptyText}>No spending data for this period</Text>
                        <Text style={styles.emptySubtext}>Start adding transactions to see your charts</Text>
                    </Animated.View>
                )}

                {/* Selected Category Details */}
                {selectedCategory !== null && data.length > 0 && data[selectedCategory] ? (
                    <Animated.View key={selectedCategory} entering={FadeIn.duration(400)} style={styles.detailsCard}>
                        <View style={styles.detailsHeader}>
                            <View style={[styles.categoryIcon, { backgroundColor: data[selectedCategory].color + '20' }]}>
                                <Ionicons name={data[selectedCategory].icon} size={24} color={data[selectedCategory].color} />
                            </View>
                            <View style={styles.detailsInfo}>
                                <Text style={styles.categoryName}>{data[selectedCategory].name}</Text>
                                <Text style={styles.categoryAmount}>{formatCurrency(data[selectedCategory].amount)}</Text>
                            </View>
                            <View style={styles.percentageContainer}>
                                <Text style={styles.percentageValue}>{data[selectedCategory].percentage.toFixed(0)}%</Text>
                                <Text style={styles.percentageLabel}>of total</Text>
                            </View>
                        </View>
                    </Animated.View>
                ) : null}

                {/* Category Breakdown */}
                {data.length > 0 && data[0].amount > 0 && (
                    <Animated.View entering={FadeInUp.delay(700).duration(600)} style={styles.breakdownSection}>
                        <Text style={styles.sectionTitle}>Category Breakdown</Text>
                        {data.map((category, index) => (
                            <TouchableOpacity
                                key={index}
                                style={styles.breakdownItem}
                                onPress={() => setSelectedCategory(index)}
                            >
                                <View style={styles.breakdownLeft}>
                                    <View style={[styles.colorIndicator, { backgroundColor: category.color }]} />
                                    <Ionicons name={category.icon} size={20} color={Colors.text} style={{ marginRight: 12 }} />
                                    <Text style={styles.breakdownName}>{category.name}</Text>
                                </View>
                                <View style={styles.breakdownRight}>
                                    <Text style={styles.breakdownAmount}>{formatCurrency(category.amount)}</Text>
                                    <View style={styles.breakdownBar}>
                                        <View 
                                            style={[
                                                styles.breakdownBarFill, 
                                                { width: `${category.percentage}%`, backgroundColor: category.color }
                                            ]} 
                                        />
                                    </View>
                                </View>
                            </TouchableOpacity>
                        ))}
                    </Animated.View>
                )}
            </ScrollView>

            {/* Floating Action Buttons */}
            <TouchableOpacity
                style={styles.fabChat}
                onPress={() => router.push('/chat')}
                activeOpacity={0.8}
            >
                <Ionicons name="chatbubble-ellipses" size={24} color="#000" />
            </TouchableOpacity>

            <TouchableOpacity
                style={styles.fab}
                onPress={() => router.push('/add-transaction')}
                activeOpacity={0.8}
            >
                <Ionicons name="add" size={28} color="#000" />
            </TouchableOpacity>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.background,
        paddingTop: 60,
    },
    centerContent: {
        justifyContent: 'center',
        alignItems: 'center',
    },
    loadingText: {
        color: Colors.textDim,
        marginTop: 16,
        fontSize: 14,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 24,
        marginBottom: 24,
    },
    headerLeft: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    headerRight: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    headerTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: Colors.text,
    },
    filterButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: Colors.card,
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 1,
        borderColor: Colors.border,
    },
    profileButton: {
        padding: 4,
    },
    tabsContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        marginBottom: 24,
        backgroundColor: Colors.card,
        marginHorizontal: 24,
        borderRadius: 24,
        padding: 4,
        borderWidth: 1,
        borderColor: Colors.border,
    },
    tab: {
        paddingVertical: 10,
        paddingHorizontal: 24,
        borderRadius: 20,
    },
    activeTab: {
        backgroundColor: Colors.primary,
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.3,
        shadowRadius: 4,
        elevation: 3,
    },
    tabText: {
        color: Colors.textDim,
        fontWeight: '500',
        fontSize: 14,
    },
    activeTabText: {
        color: '#000',
        fontWeight: '600',
        fontSize: 14,
    },
    content: {
        paddingHorizontal: 24,
        paddingBottom: 100,
    },
    totalCard: {
        backgroundColor: Colors.card,
        borderRadius: 24,
        padding: 20,
        marginBottom: 24,
        borderWidth: 1,
        borderColor: Colors.border,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 8,
        elevation: 4,
    },
    totalCardHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: 8,
    },
    totalLabel: {
        fontSize: 14,
        color: Colors.textDim,
        marginBottom: 8,
    },
    totalAmount: {
        fontSize: 32,
        fontWeight: 'bold',
        color: Colors.text,
    },
    trendBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: Colors.success + '20',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 16,
        gap: 4,
    },
    trendText: {
        fontSize: 12,
        fontWeight: '600',
    },
    comparisonText: {
        fontSize: 12,
        color: Colors.textDim,
    },
    chartContainer: {
        backgroundColor: Colors.card,
        borderRadius: 24,
        padding: 16,
        marginBottom: 24,
        borderWidth: 1,
        borderColor: Colors.border,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 8,
        elevation: 4,
    },
    emptyChart: {
        backgroundColor: Colors.card,
        borderRadius: 24,
        padding: 40,
        marginBottom: 24,
        borderWidth: 1,
        borderColor: Colors.border,
        alignItems: 'center',
        justifyContent: 'center',
    },
    emptyText: {
        color: Colors.text,
        fontSize: 16,
        fontWeight: '600',
        marginTop: 16,
    },
    emptySubtext: {
        color: Colors.textDim,
        fontSize: 14,
        marginTop: 8,
        textAlign: 'center',
    },
    detailsCard: {
        backgroundColor: Colors.card,
        borderRadius: 24,
        padding: 20,
        marginBottom: 24,
        borderWidth: 1,
        borderColor: Colors.border,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 8,
        elevation: 4,
    },
    detailsHeader: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    categoryIcon: {
        width: 48,
        height: 48,
        borderRadius: 24,
        justifyContent: 'center',
        alignItems: 'center',
        marginRight: 16,
    },
    detailsInfo: {
        flex: 1,
    },
    categoryName: {
        fontSize: 16,
        color: Colors.text,
        fontWeight: '600',
        marginBottom: 4,
    },
    categoryAmount: {
        fontSize: 24,
        fontWeight: 'bold',
        color: Colors.text,
    },
    percentageContainer: {
        alignItems: 'flex-end',
    },
    percentageValue: {
        fontSize: 20,
        fontWeight: 'bold',
        color: Colors.primary,
    },
    percentageLabel: {
        fontSize: 11,
        color: Colors.textDim,
    },
    breakdownSection: {
        marginBottom: 24,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: Colors.text,
        marginBottom: 16,
    },
    breakdownItem: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: Colors.card,
        padding: 16,
        borderRadius: 16,
        marginBottom: 12,
        borderWidth: 1,
        borderColor: Colors.border,
    },
    breakdownLeft: {
        flexDirection: 'row',
        alignItems: 'center',
        flex: 1,
    },
    colorIndicator: {
        width: 4,
        height: 32,
        borderRadius: 2,
        marginRight: 12,
    },
    breakdownName: {
        fontSize: 15,
        color: Colors.text,
        fontWeight: '500',
    },
    breakdownRight: {
        alignItems: 'flex-end',
        flex: 1,
    },
    breakdownAmount: {
        fontSize: 16,
        fontWeight: '600',
        color: Colors.text,
        marginBottom: 6,
    },
    breakdownBar: {
        width: 100,
        height: 6,
        backgroundColor: Colors.border,
        borderRadius: 3,
        overflow: 'hidden',
    },
    breakdownBarFill: {
        height: '100%',
        borderRadius: 3,
    },
    fabChat: {
        position: 'absolute',
        bottom: 170,
        right: 20,
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: Colors.primary,
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.3,
        shadowRadius: 5,
        elevation: 5,
    },
    fab: {
        position: 'absolute',
        bottom: 100,
        right: 20,
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: Colors.primary,
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.3,
        shadowRadius: 5,
        elevation: 5,
    },
});
