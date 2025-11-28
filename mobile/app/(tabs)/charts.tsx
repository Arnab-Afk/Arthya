import React, { useState } from 'react';
import { StyleSheet, View, Text, ScrollView, TouchableOpacity } from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { Svg, Rect, Defs, LinearGradient as SvgLinearGradient, Stop, Text as SvgText, Circle, Line } from 'react-native-svg';
import { Dimensions } from 'react-native';
import Animated, { FadeInDown, FadeInUp, FadeIn } from 'react-native-reanimated';

const { width } = Dimensions.get('window');

type TimePeriod = 'weekly' | 'monthly' | 'annually';
type Category = 'Food' | 'Shopping' | 'Transport' | 'Bills' | 'Entertainment';

interface CategoryData {
    name: Category;
    amount: number;
    percentage: number;
    color: string;
    icon: keyof typeof Ionicons.glyphMap;
    trend: number;
}

const chartData: Record<TimePeriod, CategoryData[]> = {
    weekly: [
        { name: 'Food', amount: 320, percentage: 35, color: '#9FE8AE', icon: 'restaurant', trend: -5 },
        { name: 'Shopping', amount: 280, percentage: 30, color: '#D4B483', icon: 'cart', trend: 12 },
        { name: 'Transport', amount: 150, percentage: 16, color: '#FF9500', icon: 'car', trend: -3 },
        { name: 'Bills', amount: 100, percentage: 11, color: '#64D2FF', icon: 'receipt', trend: 0 },
        { name: 'Entertainment', amount: 75, percentage: 8, color: '#FF453A', icon: 'game-controller', trend: 8 },
    ],
    monthly: [
        { name: 'Food', amount: 1450, percentage: 32, color: '#9FE8AE', icon: 'restaurant', trend: -8 },
        { name: 'Shopping', amount: 1280, percentage: 28, color: '#D4B483', icon: 'cart', trend: 15 },
        { name: 'Transport', amount: 780, percentage: 17, color: '#FF9500', icon: 'car', trend: -2 },
        { name: 'Bills', amount: 650, percentage: 14, color: '#64D2FF', icon: 'receipt', trend: 0 },
        { name: 'Entertainment', amount: 390, percentage: 9, color: '#FF453A', icon: 'game-controller', trend: 5 },
    ],
    annually: [
        { name: 'Food', amount: 16800, percentage: 30, color: '#9FE8AE', icon: 'restaurant', trend: -10 },
        { name: 'Shopping', amount: 15600, percentage: 28, color: '#D4B483', icon: 'cart', trend: 18 },
        { name: 'Transport', amount: 10400, percentage: 19, color: '#FF9500', icon: 'car', trend: -5 },
        { name: 'Bills', amount: 7800, percentage: 14, color: '#64D2FF', icon: 'receipt', trend: 2 },
        { name: 'Entertainment', amount: 5200, percentage: 9, color: '#FF453A', icon: 'game-controller', trend: 12 },
    ],
};

export default function ChartsScreen() {
    const [selectedPeriod, setSelectedPeriod] = useState<TimePeriod>('monthly');
    const [selectedCategory, setSelectedCategory] = useState<number>(0);
    const data = chartData[selectedPeriod];

    const maxAmount = Math.max(...data.map(d => d.amount));
    const chartHeight = 280;
    const barWidth = 56;
    const barSpacing = ((width - 48) - (barWidth * data.length)) / (data.length + 1);

    return (
        <View style={styles.container}>
            <Animated.View entering={FadeInDown.delay(100).duration(500)} style={styles.header}>
                <View style={styles.headerLeft}>
                    <Ionicons name="stats-chart" size={28} color={Colors.primary} />
                    <Text style={styles.headerTitle}>Spending Analysis</Text>
                </View>
                <TouchableOpacity style={styles.filterButton}>
                    <Ionicons name="filter" size={20} color={Colors.text} />
                </TouchableOpacity>
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

            <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
                {/* Total Spending Card */}
                <Animated.View entering={FadeInUp.delay(300).duration(600)} style={styles.totalCard}>
                    <View style={styles.totalCardHeader}>
                        <View>
                            <Text style={styles.totalLabel}>Total Spending</Text>
                            <Text style={styles.totalAmount}>${data.reduce((sum, d) => sum + d.amount, 0).toLocaleString()}</Text>
                        </View>
                        <View style={styles.trendBadge}>
                            <Ionicons name="trending-down" size={16} color={Colors.success} />
                            <Text style={styles.trendText}>-5.2%</Text>
                        </View>
                    </View>
                    <Text style={styles.comparisonText}>vs last {selectedPeriod === 'weekly' ? 'week' : selectedPeriod === 'monthly' ? 'month' : 'year'}</Text>
                </Animated.View>

                {/* Interactive Chart */}
                <Animated.View entering={FadeInUp.delay(500).duration(800)} style={styles.chartContainer}>
                    <Svg height={chartHeight + 60} width={width - 48}>
                        <Defs>
                            {data.map((category, index) => (
                                <SvgLinearGradient key={index} id={`grad${index}`} x1="0" y1="0" x2="0" y2="1">
                                    <Stop offset="0" stopColor={category.color} stopOpacity="1" />
                                    <Stop offset="1" stopColor={category.color} stopOpacity="0.3" />
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
                                    x2={width - 48}
                                    y2={y}
                                    stroke={Colors.border}
                                    strokeWidth="1"
                                    strokeDasharray="4,4"
                                />
                            );
                        })}

                        {/* Bars */}
                        {data.map((category, index) => {
                            const barHeight = (category.amount / maxAmount) * (chartHeight - 40);
                            const x = barSpacing + (index * (barWidth + barSpacing));
                            const y = chartHeight - barHeight;
                            const isSelected = index === selectedCategory;

                            return (
                                <React.Fragment key={index}>
                                    {/* Background bar */}
                                    <Rect
                                        x={x}
                                        y={20}
                                        width={barWidth}
                                        height={chartHeight - 20}
                                        rx="28"
                                        fill={Colors.card}
                                    />
                                    {/* Value bar */}
                                    <Rect
                                        x={x}
                                        y={y}
                                        width={barWidth}
                                        height={barHeight}
                                        rx="28"
                                        fill={`url(#grad${index})`}
                                        onPress={() => setSelectedCategory(index)}
                                    />
                                    {/* Selection indicator */}
                                    {isSelected && (
                                        <>
                                            <Circle
                                                cx={x + barWidth / 2}
                                                cy={y + barHeight / 2}
                                                r="30"
                                                stroke={category.color}
                                                strokeWidth="2"
                                                strokeDasharray="5 5"
                                                fill="none"
                                            />
                                            <Circle
                                                cx={x + barWidth / 2}
                                                cy={y + barHeight / 2}
                                                r="8"
                                                fill={category.color}
                                            />
                                        </>
                                    )}
                                    {/* Labels */}
                                    <SvgText
                                        x={x + barWidth / 2}
                                        y={chartHeight + 25}
                                        fill={isSelected ? Colors.text : Colors.textDim}
                                        fontSize="12"
                                        fontWeight={isSelected ? '600' : '400'}
                                        textAnchor="middle"
                                    >
                                        {category.name}
                                    </SvgText>
                                </React.Fragment>
                            );
                        })}
                    </Svg>
                </Animated.View>

                {/* Selected Category Details */}
                <Animated.View key={selectedCategory} entering={FadeIn.duration(400)} style={styles.detailsCard}>
                    <View style={styles.detailsHeader}>
                        <View style={[styles.categoryIcon, { backgroundColor: data[selectedCategory].color + '20' }]}>
                            <Ionicons name={data[selectedCategory].icon} size={24} color={data[selectedCategory].color} />
                        </View>
                        <View style={styles.detailsInfo}>
                            <Text style={styles.categoryName}>{data[selectedCategory].name}</Text>
                            <Text style={styles.categoryAmount}>${data[selectedCategory].amount.toLocaleString()}</Text>
                        </View>
                        <View style={styles.percentageContainer}>
                            <Text style={styles.percentageValue}>{data[selectedCategory].percentage}%</Text>
                            <Text style={styles.percentageLabel}>of total</Text>
                        </View>
                    </View>
                    <View style={styles.trendContainer}>
                        <Ionicons 
                            name={data[selectedCategory].trend > 0 ? "trending-up" : data[selectedCategory].trend < 0 ? "trending-down" : "remove"} 
                            size={16} 
                            color={data[selectedCategory].trend > 0 ? Colors.danger : data[selectedCategory].trend < 0 ? Colors.success : Colors.textDim} 
                        />
                        <Text style={[
                            styles.trendValue,
                            { color: data[selectedCategory].trend > 0 ? Colors.danger : data[selectedCategory].trend < 0 ? Colors.success : Colors.textDim }
                        ]}>
                            {data[selectedCategory].trend > 0 ? '+' : ''}{data[selectedCategory].trend}% from last period
                        </Text>
                    </View>
                </Animated.View>

                {/* Category Breakdown */}
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
                                <Text style={styles.breakdownAmount}>${category.amount.toLocaleString()}</Text>
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
            </ScrollView>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.background,
        paddingTop: 60,
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
        paddingBottom: 40,
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
        color: Colors.success,
        fontSize: 13,
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
        marginBottom: 16,
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
    trendContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        paddingTop: 12,
        borderTopWidth: 1,
        borderTopColor: Colors.border,
    },
    trendValue: {
        fontSize: 13,
        fontWeight: '500',
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
});
