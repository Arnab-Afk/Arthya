import React, { useState } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, TextInput, Alert, ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { FadeInDown, ZoomIn } from 'react-native-reanimated';
import { useAuth } from '@/contexts/AuthContext';

export default function LoginScreen() {
    const router = useRouter();
    const { login, register, isAuthenticated } = useAuth();
    const [isLogin, setIsLogin] = useState(true);
    const [loading, setLoading] = useState(false);
    
    // Form state
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [occupation, setOccupation] = useState<'driver' | 'freelancer' | 'hybrid' | 'other'>('freelancer');

    // Redirect if already authenticated
    React.useEffect(() => {
        if (isAuthenticated) {
            router.replace('/(tabs)');
        }
    }, [isAuthenticated]);

    const handleSubmit = async () => {
        if (!email || !password) {
            Alert.alert('Error', 'Please fill in all fields');
            return;
        }

        if (!isLogin && !name) {
            Alert.alert('Error', 'Please enter your name');
            return;
        }

        setLoading(true);
        try {
            if (isLogin) {
                await login({ email, password });
                router.replace('/(tabs)');
            } else {
                await register({ name, email, password, occupation });
                router.replace('/(tabs)');
            }
        } catch (error: any) {
            Alert.alert('Error', error.message || 'Authentication failed');
        } finally {
            setLoading(false);
        }
    };

    const toggleMode = () => {
        setIsLogin(!isLogin);
        setName('');
        setEmail('');
        setPassword('');
    };

    return (
        <KeyboardAvoidingView 
            style={styles.container}
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        >
            <LinearGradient
                colors={['#000000', '#1C1C1E']}
                style={styles.background}
            />

            <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
                <View style={styles.logoContainer}>
                    <Animated.View entering={ZoomIn.delay(200).duration(600)} style={styles.logo}>
                        <Ionicons name="wallet" size={40} color={Colors.primary} />
                    </Animated.View>
                    <Animated.Text entering={FadeInDown.delay(400).duration(600)} style={styles.appName}>Arthya</Animated.Text>
                    <Animated.Text entering={FadeInDown.delay(500).duration(600)} style={styles.tagline}>
                        {isLogin ? 'Welcome Back' : 'Create Your Account'}
                    </Animated.Text>
                </View>

                <View style={styles.formContainer}>
                    {!isLogin && (
                        <Animated.View entering={FadeInDown.delay(600).duration(600)} style={styles.inputContainer}>
                            <Ionicons name="person-outline" size={20} color={Colors.textDim} style={styles.inputIcon} />
                            <TextInput
                                style={styles.input}
                                placeholder="Full Name"
                                placeholderTextColor={Colors.textDim}
                                value={name}
                                onChangeText={setName}
                                autoCapitalize="words"
                            />
                        </Animated.View>
                    )}

                    <Animated.View entering={FadeInDown.delay(700).duration(600)} style={styles.inputContainer}>
                        <Ionicons name="mail-outline" size={20} color={Colors.textDim} style={styles.inputIcon} />
                        <TextInput
                            style={styles.input}
                            placeholder="Email"
                            placeholderTextColor={Colors.textDim}
                            value={email}
                            onChangeText={setEmail}
                            keyboardType="email-address"
                            autoCapitalize="none"
                        />
                    </Animated.View>

                    <Animated.View entering={FadeInDown.delay(800).duration(600)} style={styles.inputContainer}>
                        <Ionicons name="lock-closed-outline" size={20} color={Colors.textDim} style={styles.inputIcon} />
                        <TextInput
                            style={styles.input}
                            placeholder="Password"
                            placeholderTextColor={Colors.textDim}
                            value={password}
                            onChangeText={setPassword}
                            secureTextEntry
                        />
                    </Animated.View>

                    {!isLogin && (
                        <Animated.View entering={FadeInDown.delay(900).duration(600)} style={styles.occupationContainer}>
                            <Text style={styles.occupationLabel}>I am a:</Text>
                            <View style={styles.occupationButtons}>
                                {(['driver', 'freelancer', 'hybrid', 'other'] as const).map((type) => (
                                    <TouchableOpacity
                                        key={type}
                                        style={[
                                            styles.occupationButton,
                                            occupation === type && styles.occupationButtonActive
                                        ]}
                                        onPress={() => setOccupation(type)}
                                    >
                                        <Text style={[
                                            styles.occupationButtonText,
                                            occupation === type && styles.occupationButtonTextActive
                                        ]}>
                                            {type.charAt(0).toUpperCase() + type.slice(1)}
                                        </Text>
                                    </TouchableOpacity>
                                ))}
                            </View>
                        </Animated.View>
                    )}

                    <Animated.View entering={FadeInDown.delay(1000).duration(600)} style={{ width: '100%' }}>
                        <TouchableOpacity 
                            style={[styles.submitButton, loading && styles.submitButtonDisabled]} 
                            onPress={handleSubmit}
                            disabled={loading}
                        >
                            {loading ? (
                                <ActivityIndicator color="#fff" />
                            ) : (
                                <Text style={styles.submitButtonText}>
                                    {isLogin ? 'Login' : 'Create Account'}
                                </Text>
                            )}
                        </TouchableOpacity>
                    </Animated.View>

                    <Animated.View entering={FadeInDown.delay(1100).duration(600)} style={styles.toggleContainer}>
                        <Text style={styles.toggleText}>
                            {isLogin ? "Don't have an account? " : "Already have an account? "}
                        </Text>
                        <TouchableOpacity onPress={toggleMode}>
                            <Text style={styles.toggleLink}>
                                {isLogin ? 'Sign Up' : 'Login'}
                            </Text>
                        </TouchableOpacity>
                    </Animated.View>
                </View>
            </ScrollView>
        </KeyboardAvoidingView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.background,
    },
    background: {
        position: 'absolute',
        left: 0,
        right: 0,
        top: 0,
        bottom: 0,
    },
    scrollContent: {
        flexGrow: 1,
        justifyContent: 'center',
        padding: 24,
        paddingTop: 60,
    },
    logoContainer: {
        alignItems: 'center',
        marginBottom: 48,
    },
    logo: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: '#333',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 20,
        borderWidth: 2,
        borderColor: Colors.primary,
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
        elevation: 8,
    },
    appName: {
        fontSize: 32,
        fontWeight: 'bold',
        color: Colors.text,
        letterSpacing: 2,
    },
    tagline: {
        color: Colors.textDim,
        marginTop: 8,
        fontSize: 14,
    },
    formContainer: {
        width: '100%',
        gap: 16,
    },
    inputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: Colors.card,
        borderRadius: 12,
        borderWidth: 1,
        borderColor: Colors.border,
        paddingHorizontal: 16,
        height: 56,
    },
    inputIcon: {
        marginRight: 12,
    },
    input: {
        flex: 1,
        color: Colors.text,
        fontSize: 16,
    },
    occupationContainer: {
        marginTop: 8,
    },
    occupationLabel: {
        color: Colors.textDim,
        fontSize: 14,
        marginBottom: 12,
    },
    occupationButtons: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 8,
    },
    occupationButton: {
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: 20,
        backgroundColor: Colors.card,
        borderWidth: 1,
        borderColor: Colors.border,
    },
    occupationButtonActive: {
        backgroundColor: Colors.primary,
        borderColor: Colors.primary,
    },
    occupationButtonText: {
        color: Colors.textDim,
        fontSize: 14,
    },
    occupationButtonTextActive: {
        color: '#fff',
        fontWeight: '600',
    },
    submitButton: {
        width: '100%',
        paddingVertical: 16,
        borderRadius: 12,
        backgroundColor: Colors.primary,
        alignItems: 'center',
        marginTop: 8,
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
        elevation: 5,
    },
    submitButtonDisabled: {
        opacity: 0.6,
    },
    submitButtonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
    },
    toggleContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        marginTop: 24,
    },
    toggleText: {
        color: Colors.textDim,
        fontSize: 14,
    },
    toggleLink: {
        color: Colors.primary,
        fontSize: 14,
        fontWeight: '600',
    },
});
